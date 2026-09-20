"""Fetch live generation and forecast data from the Data Platform and write it in scorecard format.

Connects to a Data Platform gRPC instance (default localhost:50051 -- e.g. via an SSH tunnel
to the dev environment) and, for a single location (default: uk national) and time window:

1. Fetches observed generation (via GetObservationsAsTimeseries) and writes it as a Zarr
   store matching the ocf_data_sampler generation schema:
   https://github.com/openclimatefix/ocf-data-sampler/blob/main/ocf_data_sampler/load/generation.py

2. Fetches forecasts (via StreamForecastData) for one or more forecasters and writes each as
   a Zarr store matching the format expected by the PVNet scorecard script:
   https://github.com/openclimatefix/pvnet/blob/main/scripts/scorecard/generate_scorecard.py

Example usage:
```
python fetch_from_data_platform.py \
    --location-name uk \
    --forecasters pvnet_v2 blend \
    --observer pvlive_in_day \
    --start 2026-09-20T00:00:00 \
    --end 2026-09-20T04:35:00 \
    --output-dir ./data
```
"""

import argparse
import asyncio
import datetime as dt
import os

import numpy as np
import xarray as xr
from grpclib.client import Channel
from ocf import dp

# Quantile levels the scorecard script expects, matching PVNet's standard output quantiles.
# p50 comes from `p50_fraction`; the rest come from `other_statistics_fractions`.
QUANTILE_KEYS = {
    0.02: "p02",
    0.10: "p10",
    0.25: "p25",
    0.50: None,  # handled specially via p50_fraction
    0.75: "p75",
    0.90: "p90",
    0.98: "p98",
}


async def get_client(host: str, port: int) -> tuple[Channel, dp.DataPlatformDataServiceStub]:
    """Open a gRPC channel to the Data Platform and return it with a ready-to-use client."""
    channel = Channel(host=host, port=port)
    return channel, dp.DataPlatformDataServiceStub(channel)


async def resolve_location(
    client: dp.DataPlatformDataServiceStub,
    location_name: str,
) -> dp.ListLocationsResponseLocationSummary:
    """Look up a SOLAR location by name (e.g. "uk" for the national location)."""
    resp = await client.list_locations(
        dp.ListLocationsRequest(
            location_names_filter=[location_name],
            energy_source_filter=dp.EnergySource.SOLAR,
        ),
    )
    if not resp.locations:
        raise ValueError(f"No location found matching name={location_name!r}")
    return resp.locations[0]


async def resolve_forecaster_version(
    client: dp.DataPlatformDataServiceStub,
    forecaster_name: str,
) -> str:
    """StreamForecastData requires an exact forecaster_version to be set."""
    resp = await client.list_forecasters(
        dp.ListForecastersRequest(
            forecaster_names_filter=[forecaster_name],
            latest_versions_only=True,
        ),
    )
    matches = [f for f in resp.forecasters if f.forecaster_name == forecaster_name]
    if not matches:
        raise ValueError(f"No forecaster found matching name={forecaster_name!r}")
    return matches[0].forecaster_version


async def fetch_generation(
    client: dp.DataPlatformDataServiceStub,
    location: dp.ListLocationsResponseLocationSummary,
    observer_name: str,
    start: dt.datetime,
    end: dt.datetime,
) -> xr.Dataset:
    """Fetch observed generation and format it per the ocf_data_sampler generation schema."""
    resp = await client.get_observations_as_timeseries(
        dp.GetObservationsAsTimeseriesRequest(
            location_uuid=location.location_uuid,
            observer_name=observer_name,
            energy_source=dp.EnergySource.SOLAR,
            time_window=dp.TimeWindow(start_timestamp_utc=start, end_timestamp_utc=end),
        ),
    )
    if not resp.values:
        raise ValueError(
            f"No observations returned for observer={observer_name!r} "
            f"location={location.location_name!r} window={start}-{end}",
        )

    times = np.array([v.timestamp_utc for v in resp.values], dtype="datetime64[ns]")
    generation_mw = np.array(
        [v.value_fraction * v.effective_capacity_watts / 1e6 for v in resp.values],
        dtype="float32",
    )
    capacity_mwp = np.array(
        [v.effective_capacity_watts / 1e6 for v in resp.values],
        dtype="float32",
    )

    ds = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), generation_mw[:, None]),
            "capacity_mwp": (("time_utc", "location_id"), capacity_mwp[:, None]),
        },
        coords={
            "time_utc": times,
            "location_id": np.array([0], dtype="int64"),
            "longitude": ("location_id", np.array([location.latlng.longitude], dtype="float64")),
            "latitude": ("location_id", np.array([location.latlng.latitude], dtype="float64")),
        },
    )
    return ds


async def fetch_forecast(
    client: dp.DataPlatformDataServiceStub,
    location: dp.ListLocationsResponseLocationSummary,
    forecaster_name: str,
    start: dt.datetime,
    end: dt.datetime,
) -> xr.Dataset:
    """Fetch forecast data and format it per the PVNet scorecard backtest schema."""
    forecaster_version = await resolve_forecaster_version(client, forecaster_name)
    forecaster = dp.Forecaster(
        forecaster_name=forecaster_name,
        forecaster_version=forecaster_version,
    )

    values: list[dp.ForecastDatum] = []
    request = dp.StreamForecastDataRequest(
        location_uuids=[location.location_uuid],
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(start_timestamp_utc=start, end_timestamp_utc=end),
        forecasters=[forecaster],
        include_metadata=False,
    )
    async for chunk in client.stream_forecast_data(request):
        values.extend(chunk.values)

    if not values:
        raise ValueError(
            f"No forecast values returned for forecaster={forecaster_name!r} "
            f"(v{forecaster_version}) location={location.location_name!r} "
            f"window={start}-{end}",
        )

    init_times = sorted({v.init_timestamp for v in values})
    horizons = sorted({v.horizon_mins for v in values})
    quantiles = sorted(QUANTILE_KEYS.keys())

    init_index = {t: i for i, t in enumerate(init_times)}
    horizon_index = {h: i for i, h in enumerate(horizons)}
    quantile_index = {q: i for i, q in enumerate(quantiles)}

    hindcast = np.full(
        (len(init_times), 1, len(horizons), len(quantiles)),
        np.nan,
        dtype="float64",
    )

    for v in values:
        i = init_index[v.init_timestamp]
        h = horizon_index[v.horizon_mins]
        capacity_mw = v.effective_capacity_watts / 1e6

        hindcast[i, 0, h, quantile_index[0.50]] = v.p50_fraction * capacity_mw
        for q, key in QUANTILE_KEYS.items():
            if key is not None and key in v.other_statistics_fractions:
                hindcast[i, 0, h, quantile_index[q]] = (
                    v.other_statistics_fractions[key] * capacity_mw
                )

    ds = xr.Dataset(
        data_vars={
            "hindcast": (
                ("init_time_utc", "location_id", "step", "quantile"),
                hindcast,
            ),
        },
        coords={
            "init_time_utc": np.array(init_times, dtype="datetime64[ns]"),
            "location_id": np.array([0], dtype="int64"),
            "step": np.array(horizons, dtype="timedelta64[m]").astype("timedelta64[ns]"),
            "quantile": np.array(quantiles, dtype="float64"),
        },
    )
    return ds


async def main(args: argparse.Namespace) -> None:
    """Fetch generation and forecast data for the configured location/window and write Zarrs."""
    os.makedirs(args.output_dir, exist_ok=True)

    channel, client = await get_client(args.host, args.port)
    try:
        print(f":: Resolving location {args.location_name!r}")
        location = await resolve_location(client, args.location_name)
        print(
            f"\tuuid={location.location_uuid} "
            f"capacity={location.effective_capacity_watts / 1e6:.1f} MW",
        )

        start_str = args.start.strftime("%Y%m%dT%H%M")
        end_str = args.end.strftime("%Y%m%dT%H%M")

        print(f":: Fetching generation ({args.observer}) for {args.start} -> {args.end}")
        generation_ds = await fetch_generation(
            client,
            location,
            args.observer,
            args.start,
            args.end,
        )
        generation_path = os.path.join(
            args.output_dir,
            f"{args.location_name}_generation_{start_str}_{end_str}.zarr",
        )
        generation_ds.to_zarr(generation_path, mode="w")
        print(f"\tWrote {generation_path} ({generation_ds.sizes['time_utc']} timesteps)")

        forecast_paths: dict[str, str] = {}
        for forecaster_name in args.forecasters:
            print(f":: Fetching forecast {forecaster_name!r} for {args.start} -> {args.end}")
            forecast_ds = await fetch_forecast(
                client,
                location,
                forecaster_name,
                args.start,
                args.end,
            )
            forecast_path = os.path.join(
                args.output_dir,
                f"{args.location_name}_{forecaster_name}_{start_str}_{end_str}.zarr",
            )
            forecast_ds.to_zarr(forecast_path, mode="w")
            forecast_paths[forecaster_name] = forecast_path
            print(
                f"\tWrote {forecast_path} "
                f"({forecast_ds.sizes['init_time_utc']} init times, "
                f"{forecast_ds.sizes['step']} horizons)",
            )

        print("\n:: Done. Paths for scorecard_config.yaml:")
        print(f"y_true_path: {generation_path}")
        for name, path in forecast_paths.items():
            print(f"  {name}: {path}")
    finally:
        channel.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fetch script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("DATA_PLATFORM_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DATA_PLATFORM_PORT", "50051")))
    parser.add_argument("--location-name", default="uk")
    parser.add_argument("--observer", default="pvlive_in_day")
    parser.add_argument("--forecasters", nargs="+", default=["pvnet_v2", "blend"])
    parser.add_argument(
        "--start",
        type=lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC),
        required=True,
        help="Start of the time window, UTC ISO format, e.g. 2026-09-20T00:00:00",
    )
    parser.add_argument(
        "--end",
        type=lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC),
        required=True,
        help="End of the time window, UTC ISO format, e.g. 2026-09-20T04:35:00",
    )
    parser.add_argument("--output-dir", default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
