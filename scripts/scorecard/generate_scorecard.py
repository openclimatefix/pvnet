"""A script to create a standardised scorecard from several backtests.

This script uses an HTML template and populates it with graphs and tables of chosen metrics,
generated from provided backtest files.

This script automatically generates a PDF report with core graphs and tables for comparing model
performance. It takes in backtest and ground truth files, populates an HTML template, and saves it
as a PDF. It will also save any graphs it creates as PNG files, and print out tables as it goes,
so no data is lost if the report fails to compile.

Workflow:
    1. Prepare backtests: intersect backtests by t0 and format them for processing
    2. Generate and save out plots; generate tebles
    3. Compile HTML report from template
    4. Export the HTML to a formatted PDF report


Example usage:
```
python scripts/scorecard/generate_scorecard.py
```

!Make sure all user-configured variables are set in the top section of the script.

The script expects backtest files to be in the following format (standard output of the 
backtest script):

```
<xarray.Dataset> Size: 4GB
Dimensions:        (init_time_utc: 34992, location_id: 13, step: 144,
                    quantile: 7)
Coordinates:
  * init_time_utc  (init_time_utc) datetime64[ns] 280kB 2024-01-01 ... 2024-1...
  * location_id    (location_id) int64 104B 0 1 2 3 4 5 6 7 8 9 10 11 12
  * step           (step) timedelta64[ns] 1kB 00:15:00 ... 1 days 12:00:00
  * quantile       (quantile) float64 56B 0.02 0.1 0.25 0.5 0.75 0.9 0.98
Data variables:
    hindcast       (init_time_utc, location_id, step, quantile) float64 4GB dask.array<chunksi...
```

NB: This script currently only computes metrics for a "national" (location_id=0) forecast.
"""

import base64
import os
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from jinja2 import Environment, FileSystemLoader
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
from weasyprint import HTML
import yaml

# Setting the cycler to distinguishable OCF colors.
# Comment out if not desired/replace with other colors
colors = [
        "#306BFF",  # blue
        "#FF4901",  # orange
        "#B701FF",  # purple
        "#17E58F",  # green
        "#BF4F04",  # brown
        "#10C5F7",  # light blue
        "#FC9700",  # yellow
        "#009C75",  # dark green
    ]

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    color=colors
)

# Get file's directory to be able to locate template
dir_path = os.path.dirname(os.path.realpath(__file__))


# config = yaml.load(f"{dir_path}/scorecard_config.yaml")
with open(f"{dir_path}/scorecard_config.yaml", "r") as f:
    config = yaml.safe_load(f)


#___________FUNCTIONS___________

def prep_backtests(backtest_dict: dict[str, str], y_true: xr.Dataset):
    """
    Combines backtests with ground truth and compiles into a single xr.Dataset for processing.

    The backtests are intersected by time, amended with model names, time coordintes, and error 
    calculations.

    Args:
        backtest_dict:  dictionary of backtests to be compared, with keys being model names,
                        and items backtest filepaths
        y_true:         xr.Dataset of ground truth observations
    """

    def prep_ds(ds, y_true):
        """
        Amend backtest dataset with time coordinates, ground thruths, and errors.
        
        Args:
            ds: backtest xr.Dataset
            y_true: ground truth xr.Dataset with generation and capacity data
        """
        # Remove predictions there is no ground truth for
        ds = ds.sel(init_time_utc=slice(
            y_true.time_utc.data.min(),
            y_true.time_utc.data.max()-ds.step.data.max()))
        
        # Add time coordinates
        ds = ds.assign_coords(target_datetime_utc=ds.init_time_utc+ds.step)
        ds = ds.assign_coords(t0_month=ds.init_time_utc.astype('datetime64[M]'))
        ds['step'] = ds.step.astype('timedelta64[m]').astype(int)//60

        # Attach true generation and capacities
        ds['capacity_mwp'] = y_true.capacity_mwp.sel(time_utc=ds.target_datetime_utc)
        ds['generation_mw'] = y_true.generation_mw.sel(time_utc=ds.target_datetime_utc)

        # Calculate errors
        ds['error'] = ds.hindcast - ds.generation_mw
        ds['norm_error'] = ds.error/ds.capacity_mwp
        ds['norm_abs_error'] = np.abs(ds.norm_error)
        # Add a binary flag of y_true < y_quantile that will aggregate into a fraction
        ds['q_fraction'] = (ds.hindcast > ds.generation_mw).astype(int)

        return ds.rename({'step': 'horizon_minutes'})

    backtest_ds_list = []

    # Add model name dimension to each backtest
    for model in backtest_dict.keys():
        if type(backtest_dict[model]) is list:
            ds = xr.open_mfdataset(backtest_dict[model], engine="zarr")
            ds = ds.drop_duplicates(dim='init_time_utc')
        else:
            ds = xr.open_zarr(backtest_dict[model])
            
        ds = ds.expand_dims(dict(model=[model]))
        backtest_ds_list.append(ds)

    # Intersect backtests by all dimensions except model name
    ds_all = xr.concat(backtest_ds_list, dim='model', join='inner')

    ds_all = prep_ds(ds_all, y_true)
    return ds_all.sel(location_id=0)


def plot_t0_availability(ds: xr.Dataset) -> str:
    """
    Plot histograms of t0 availability by hour and date.

    Saves the plot as .png for back-up and returns it as a string of bytes for HTML rendering.
    
    Args:
        ds: backtest xr.Dataset
    """
    fig, ax = plt.subplots(1,2, figsize=(10, 5))

    unique, counts = np.unique(pd.to_datetime(ds.init_time_utc.data).date, return_counts=True)
    ax[0].bar(unique, counts)
    ax[0].set_ylabel("# init times")
    ax[0].set_xlabel("Date")

    unique, counts = np.unique(pd.to_datetime(ds.init_time_utc.data).hour, return_counts=True)
    ax[1].bar(unique, counts)
    ax[1].set_ylabel("# init times")
    ax[1].set_xlabel("Hour")

    fig.autofmt_xdate()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plot_t0_availability = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.savefig(f"{config['output_dir']}/t0_availability.png")
    plt.close()
    return plot_t0_availability


def get_nmae_crosscomparison(df_nmae: pd.Series) -> str:
    """
    Create a table comparing NMAE across all models

    Prints the table contents and returns table as a string 
    of bytes for HTML rendering
    
    Args:
        df_nmae: pd.Series of average NMAE per model across all 
                 forecast horizons
    """
    a = df_nmae.values
    labels = df_nmae.index.values

    result = []
    for i in a:
        result.append((a-i)/a)

    nmae_crosscomparison_html = pd.DataFrame(
        data=result,
        columns=labels,
        index=labels)
    
    # Print the resutl so it's not lost if the report is not generated
    print("\nNMAE cross-comparison")
    print(nmae_crosscomparison_html)

    return nmae_crosscomparison_html.style.format(
            "{:.2%}"
        ).to_html(classes='table')


def get_horizon_nmae_plot(ds_nmae: xr.Dataset) -> tuple[str]:
    """
    Plot NMAE by forecast horizon

    Saves the plot as .png for back-up and returns it as a string of bytes for HTML rendering.
    
    Args:
        ds_nmae: xr.Dataset of NMAE per forecast horizon
    """
    _, ax = plt.subplots(1,1)
    ds_nmae.plot(hue='model', ax=ax)
    ax.set_title("NMAE by forecast_horizon")
    ax.grid()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plot_nmae = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.savefig(f"{config['output_dir']}/nmae_forecast_horizon.png")
    plt.close()

    return plot_nmae


def get_monthly_nmae_plot_and_table(ds: xr.Dataset) -> tuple[str]:
    """
    Plot NMAE by month and create a lookup table

    Saves the plot as .png for back-up and returns it as a string of bytes for HTML rendering.
    Prints out the table as a back-up and returns as a string of bytes for HTML rendering.
    
    Args:
        ds: backtest xr.Dataset
    """
    ds_nmae_month = ds.sel(
        quantile=0.5
        ).norm_abs_error.groupby('t0_month').mean('init_time_utc').mean('horizon_minutes')
    df_nmae_month = ds_nmae_month.to_dataframe().reset_index()[
        ['model', 't0_month', 'norm_abs_error']
    ]

    nmae_by_month_df = df_nmae_month.pivot(
        columns='model',
        index='t0_month',
        values='norm_abs_error'
    ).reset_index()
    nmae_by_month_df['month'] = nmae_by_month_df['t0_month'].apply(lambda x: x.strftime('%b %Y'))
    nmae_by_month_df = nmae_by_month_df.set_index('month').sort_values('t0_month').drop(
        't0_month', axis=1
    )

    print("\nNMAE by month")
    print(nmae_by_month_df)

    nmae_by_month_html = nmae_by_month_df.style.format(
        "{:.2%}"
        ).to_html(classes='table')

    _, ax = plt.subplots(1,1)
    
    nmae_by_month_df.plot(
        kind='line',
        marker='o',
        ax=ax
    )

    ax.set_ylabel("NMAE")
    ax.set_xlabel("Month")
    ax.set_title("NMAE by month")

    unique_months = nmae_by_month_df.index
    if len(unique_months) > 6:
        ax.set_xticks(range(0, len(unique_months), 2))
        ax.set_xticklabels(unique_months[::2])
    else:
        ax.set_xticks(range(len(unique_months)))
        ax.set_xticklabels(unique_months)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plot_nmae_month = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.savefig(f"{config['output_dir']}/nmae_month.png")
    plt.close()

    return plot_nmae_month, nmae_by_month_html


def plot_quantile_calibration(ds: xr.Dataset) -> tuple[str]:
    """
    Plot quantile calibration.

    Saves the plot as .png for back-up and returns it as a string of bytes for HTML rendering.
    Generates, prints, and returns lookup table of the quantile fraction values for the plot.
    
    Args:
        ds: backtest xr.Dataset
    """
    # Filter out timestamps where there is no generation
    ds_day = ds.where(ds.generation_mw>0)

    # Average out init_time and horizon to get fraction. This is grouped by model
    df = ds_day.q_fraction.groupby(
        ['model', 'quantile']
        ).mean('init_time_utc').mean('horizon_minutes').to_dataframe().reset_index(level=1)

    # Generate plot
    _, ax = plt.subplots(1,1)

    ax.plot([0.02,0.98], [0.02, 0.98], color="black", alpha=0.2)

    for model_name, group in df.groupby('model'):
        ax.plot(
            group["quantile"], 
            group["q_fraction"], 
            label=model_name,
            marker='o',
            linestyle='-',
            markersize=4
        )
    ax.set_xlabel("quantile")
    ax.set_ylabel("q_fraction")
    ax.legend(title="model")
    ax.set_yticks(ds['quantile'].data, ds['quantile'].data)
    ax.set_xticks(ds['quantile'].data, ds['quantile'].data)
    ax.set_yticks(ds['quantile'].data)
    ax.set_xticks(ds['quantile'].data)

    # Force the 2 decimal point string format ("%.2f") on both axes
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid()
    ax.set_title("Quantile calibration")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    quantile_calibration_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.savefig(f"{config['output_dir']}/quantile_calibration.png")
    plt.close()

    # Generate lookup table of values
    quantile_calibration_table = np.round(df.reset_index().pivot(
        index='quantile',
        columns='model',
        values='q_fraction'
    ), 2)

    print("\nQuantile calibration")
    print(quantile_calibration_table)

    quantile_calibration_table = quantile_calibration_table.to_html(classes="table")
    return quantile_calibration_plot, quantile_calibration_table


def create_report(backtest_dict: dict[str, str], y_true_path: str):
    """
    Create all plots and tables and compile the final report.

    Args:
        backtest_dict:  dictionary of backtests to be compared, with keys being model names,
                        and items backtest filepaths
        y_true_path:    filepath of ground truth used to compare predictions against
    """
    pbar = tqdm(total=6)

    # 1. Load and format backtests
    ds = prep_backtests(backtest_dict, y_true=xr.open_zarr(y_true_path))

    start_time, end_time = config['time_window']['start'], config['time_window']['end']
    if start_time:
        ds = ds.sel(init_time_utc=slice(config['time_window']['start'], None))
    if end_time:
        ds = ds.sel(init_time_utc=slice(None, config['time_window']['end']))

    pbar.update()
    
    # 2. Plot t0 availability
    plot_t0_availability_bytes = plot_t0_availability(ds)
    pbar.update()

    # 3. Plot NMAE
    # Load this as several functions use this data
    ds_nmae = ds.sel(quantile=0.5).norm_abs_error.mean('init_time_utc').load()

    plot_nmae_bytes = get_horizon_nmae_plot(ds_nmae)

    # Calculate average and transform into a table
    ds_nmae = ds_nmae.mean('horizon_minutes')
    df_nmae = ds_nmae.to_dataframe()[['norm_abs_error']].sort_values(by='norm_abs_error')
    
    # Print NMAE results so no data is lost if script fails later
    print("\nNMAE leaderboard")
    print(df_nmae)

    nmae_leaderboard_html = df_nmae.style.format(
            "{:.2%}"
    ).to_html(classes='table')

    nmae_crosscomparison_html = get_nmae_crosscomparison(df_nmae['norm_abs_error'])
    pbar.update()

    # 4. Get quantile calibration plot and look-up table
    plot_quantile_calibration_bytes, quantile_calibration_html = plot_quantile_calibration(ds)
    pbar.update()

    # 5. Calculate NMAE by month and plot
    plot_nmae_month, nmae_by_month_html = get_monthly_nmae_plot_and_table(ds)
    pbar.update()

    # 6. Render report
    env = Environment(loader=FileSystemLoader(f"{dir_path}"))
    template = env.get_template("scorecard_template.html")

    html_out = template.render(
        model_names=", ".join(backtest_dict.keys()),
        time_period_min=pd.to_datetime(
                        str(ds.init_time_utc.min().values)
                        ).strftime('%Y-%m-%d %H:%M'),
        time_period_max=pd.to_datetime(
                        str(ds.init_time_utc.max().values)
                        ).strftime('%Y-%m-%d %H:%M'),
        plot_t0_availability=plot_t0_availability_bytes,
        plot_nmae=plot_nmae_bytes,
        nmae_leaderboard_html=nmae_leaderboard_html,
        nmae_crosscomparison_html=nmae_crosscomparison_html,
        plot_quantile_calibration=plot_quantile_calibration_bytes,
        quantile_calibration_html=quantile_calibration_html,
        model_paths=backtest_dict,
        y_true_path=y_true_path,
        nmae_by_month_html=nmae_by_month_html,
        plot_nmae_month=plot_nmae_month,
    )

    output_name = f"""{config['output_dir']}/scorecard_{'_'.join(backtest_dict.keys())}_
                        {
                        pd.to_datetime(
                            str(ds.init_time_utc.min().values)
                            ).strftime('%Y%m%dT%H%M')}_{pd.to_datetime(
                            str(ds.init_time_utc.max().values)
                            ).strftime('%Y%m%dT%H%M')
                        }.pdf"""

    HTML(string=html_out).write_pdf(output_name)

    pbar.update()
    pbar.close()
    print(f"\nReport saved as {output_name}")


if __name__ == "__main__":
    os.makedirs(config['output_dir'], exist_ok=False)
    create_report(
        backtest_dict=config['input data']['backtest_dict'],
        y_true_path=config['input data']['y_true_path']
    )
