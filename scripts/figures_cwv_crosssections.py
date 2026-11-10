# %%
"""CWV(lat) cross-sections from HAMP during PERCUSION.

1. Mean CWV(lat) for the eastern and western part of the campaign
2. Individual flights: CWV and IWP + LWP per flight
3. Gradient of CWV with latitude at different CWV thresholds (median over flights)
"""

import math

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xarray as xr
from orcestra import get_flight_segments
from scipy.ndimage import gaussian_filter1d
from xarray.groupers import BinGrouper, UniqueGrouper

HAMP_CID = "ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y"

PERIODS = {
    "west": ("2024-09-05", "2024-09-28"),
    "east": ("2024-08-11", "2024-09-05"),
}

CWV_EDGE = 48  # mm


# %% --- helper functions ---------------------------------------------------------


def load_hamp():
    """HAMP data with IWP + LWP (kg m-2) and lat / day as coordinates."""
    ds = xr.open_dataset(HAMP_CID, engine="zarr")
    ds["TILWP"] = (ds["TLWP"] + ds["IWP"]) / 1000.0  # g m-2 -> kg m-2
    ds["TILWP"].attrs = {"long_name": "IWP + LWP", "units": "kg m-2"}
    return ds.assign_coords(lat=ds["lat"], day=ds.time.dt.floor("D"))


def bin_by_lat(data, dx, min_count, by_day=False, lat_min=-2, lat_max=18):
    """Mean in latitude bins of width dx (optionally per day).

    Bins with fewer than min_count values are set to NaN.
    The result has the dimension 'lat_bins' (bin centres) and, if by_day, 'day'.
    """
    edges = np.arange(lat_min, lat_max, dx)
    centres = (edges[:-1] + edges[1:]) / 2

    groupers = {"lat": BinGrouper(bins=edges, labels=centres)}
    if by_day:
        groupers["day"] = UniqueGrouper()

    grouped = data.groupby(**groupers)
    return grouped.mean().where(grouped.count() >= min_count)


def smooth_lat(da, sigma_deg, dx):
    """Gaussian smoothing along lat_bins that ignores NaNs."""
    axis = da.get_axis_num("lat_bins")
    values = da.values
    valid = np.isfinite(values)

    sigma = sigma_deg / dx  # in bins
    num = gaussian_filter1d(
        np.where(valid, values, 0.0), sigma, axis=axis, mode="constant"
    )
    den = gaussian_filter1d(valid.astype(float), sigma, axis=axis, mode="constant")

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = num / den
    smoothed[den < 0.3] = np.nan  # too little data within the kernel
    return da.copy(data=smoothed)


def threshold_crossings(cwv, threshold):
    """Latitudes (bin centres) where cwv crosses the threshold."""
    cwv = cwv.dropna("lat_bins")
    above = cwv.values > threshold
    idx = np.where(above[:-1] != above[1:])[0]  # bin before each crossing
    return cwv.lat_bins.values[idx]


def steepest_slope(cwv, lat0, delta_lat, min_points):
    """Steepest straight-line slope (mm per degree) around lat0.

    Fits lines to three windows - south of lat0, north of lat0 and centred
    on lat0 - and returns the slope with the largest absolute value.
    Returns NaN if no window has at least min_points valid values.
    """
    lat = cwv.lat_bins.values
    values = cwv.values
    windows = [
        (lat0 - delta_lat, lat0),
        (lat0, lat0 + delta_lat),
        (lat0 - delta_lat / 2, lat0 + delta_lat / 2),
    ]

    best = np.nan
    for lo, hi in windows:
        in_window = (lat >= lo) & (lat <= hi) & np.isfinite(values)
        if in_window.sum() < min_points:
            continue
        slope = np.polyfit(lat[in_window], values[in_window], deg=1)[0]
        if np.isnan(best) or abs(slope) > abs(best):
            best = slope
    return best


def slopes_at_threshold(cwv_daily, threshold, delta_lat, dx, min_valid_deg=1.0):
    """Table with one row per day and threshold crossing: day, lat, slope."""
    rows = []
    for day in cwv_daily.day.values:
        cwv = cwv_daily.sel(day=day)
        for lat0 in threshold_crossings(cwv, threshold):
            rows.append(
                {
                    "day": pd.Timestamp(day),
                    "threshold": threshold,
                    "lat": lat0,
                    "slope": steepest_slope(
                        cwv, lat0, delta_lat, min_points=int(min_valid_deg / dx)
                    ),
                }
            )
    return pd.DataFrame(rows)


# %% --- load data ----------------------------------------------------------------

hamp = load_hamp()

flights = get_flight_segments()["HALO"]
flight_ids = {pd.Timestamp(info["date"]): fid for fid, info in flights.items()}

# %% --- figure 1: mean CWV(lat), east and west -------------------------------------

fig, axes = plt.subplots(
    1, 2, figsize=(9, 3), constrained_layout=True, sharey=True, sharex=True
)

clrs = {"IWV": "tab:blue", "TILWP": "tab:orange"}

mean_clrs = {"west": "#a33904", "east": "#045e94"}
day_cmaps = {"west": "YlOrBr", "east": "PuBu"}

for i_period, [name, (start, end)] in enumerate(PERIODS.items()):
    for i_var, var in enumerate(["IWV"]):  # , "TILWP"

        var_data = hamp[var].sel(time=slice(start, end))

        sigma_deg = 0.2
        dx = 0.2
        sigma_bins = int(sigma_deg / dx)

        mean = bin_by_lat(var_data, dx=dx, min_count=10)
        mean = mean.copy(
            data=gaussian_filter1d(
                mean.values, sigma=sigma_bins, axis=0, mode="nearest"
            )
        )

        daily = bin_by_lat(var_data, dx=dx, min_count=2, by_day=True)
        daily = daily.copy(
            data=gaussian_filter1d(
                daily.values, sigma=sigma_bins, axis=0, mode="nearest"
            )
        )

        plt.sca(axes[i_period])
        # plt.title(name)

        plt.plot(
            mean.lat_bins,
            mean,
            color=mean_clrs[name],
            linewidth=2,
            label="mean",
            zorder=20,
        )

        cmap = plt.get_cmap(day_cmaps[name])
        day_colors = cmap(np.linspace(0.35, 0.85, daily.sizes["day"]))

        for i_day, day in enumerate(daily.day.values):
            plt.plot(
                daily.sel(day=day).lat_bins,
                daily.sel(day=day),
                color=day_colors[i_day],
                zorder=10,
                alpha=1,
                linewidth=0.5,
            )

        plt.xlim(2, 18)


axes[0].set_ylabel("CWV / mm")

for ax in axes:
    ax.set_xlabel("latitude / °N")
    ax.axhline(CWV_EDGE, color="k", ls="--", lw=0.8)

axes[1].tick_params(axis="y", left=False)
sns.despine(trim=True)

# %% --- figure 2: individual flights ---------------------------------------------

flights_binned = bin_by_lat(hamp[["IWV", "TILWP"]], dx=0.2, min_count=100, by_day=True)


def plot_flights(binned, title, ncols=4):
    """One panel per flight: CWV (left axis) and IWP + LWP (right axis)."""
    days = binned.day.values
    nrows = math.ceil(len(days) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.4 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    for i, (ax, day) in enumerate(zip(axes.flat, days)):
        flight = binned.sel(day=day)
        ax.plot(flight.lat_bins, flight.IWV, color="tab:blue")
        ax.axhline(CWV_EDGE, color="tab:blue", ls="--", lw=0.8)
        ax.set_ylim(35, 68)

        twin = ax.twinx()
        twin.plot(flight.lat_bins, flight.TILWP, color="tab:orange")
        twin.set_ylim(0, 10)

        last_in_row = (i % ncols == ncols - 1) or (i == len(days) - 1)
        if last_in_row:
            twin.set_ylabel("IWP + LWP / kg m$^{-2}$", color="tab:orange")
        else:
            twin.tick_params(
                labelright=False
            )  # optional: tick labels only on the right edge too

        ax.set_title(flight_ids.get(pd.Timestamp(day), str(day)[:10]), fontsize=9)

    for ax in axes.flat[len(days) :]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("latitude / °N")
    for ax in axes[:, 0]:
        ax.set_ylabel("CWV / mm", color="tab:blue")
    fig.suptitle(title)
    sns.despine(fig=fig, trim=True)
    return fig


for name, (start, end) in PERIODS.items():
    plot_flights(flights_binned.sel(day=slice(start, end)), title=name)

# %% --- figure 3: CWV gradient at different thresholds ----------------------------

dx_fine = 0.05
cwv_daily = bin_by_lat(hamp["IWV"], dx=dx_fine, min_count=10, by_day=True)
cwv_smooth = smooth_lat(cwv_daily, sigma_deg=0.25, dx=dx_fine)

thresholds = np.arange(45, 63, 1)
slopes = pd.concat(
    [slopes_at_threshold(cwv_smooth, t, delta_lat=1.0, dx=dx_fine) for t in thresholds],
    ignore_index=True,
)
slopes["abs_slope"] = slopes["slope"].abs()

median_slope = slopes.groupby("threshold")["abs_slope"].median()

fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
ax.plot(median_slope.index, median_slope.values, "o-")
ax.set_xlabel("CWV threshold / mm")
ax.set_ylabel("median |dCWV/dlat| / mm per °")
sns.despine(trim=True)

# %%
