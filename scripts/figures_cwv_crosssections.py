# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import percusion
from percusion import utils
import pandas as pd
import math

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]
from orcestra import get_flight_segments
import xarray as xr

ds_hamp = xr.open_dataset(
    "ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y", engine="zarr"
)

ds_hamp["TILWP"] = ds_hamp["TLWP"] + ds_hamp["IWP"]
ds_hamp["TILWP"].attrs["long_name"] = "Total Ice and Liquid Water Path"
ds_hamp["TILWP"].attrs["units"] = "g m-2"

flights = get_flight_segments()["HALO"]


# --- settings ---------------------------------------------------------------

east = True  # True for the eastern part (Sal), False for the western part (Barbados)

start_date = "2024-08-11" if east else "2024-09-07"
end_date = "2024-09-05" if east else "2024-09-28"

variables = ["IWV", "TILWP"]  # first one goes on the left axis, the others on the right

plot_settings = {
    "IWV": {
        "label": "CWV / mm",
        "color": "tab:blue",
        "light": "lightsteelblue",
        "ref": 48,
        "ref_label": "CWV = 48 mm",
    },
    "TILWP": {
        "label": "TILWP / kg m$^{-2}$",
        "color": "tab:orange",
        "light": "navajowhite",
    },
}

lat_bins = np.arange(-2, 18, 0.25)
lat_bin_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
min_count_mean = 3600
min_count_flight = 100


def bin_by_lat(ds, variables, bins, labels, min_count):
    """Mean of each variable in latitude bins, masked per variable by its own count."""
    grouped = ds[variables].groupby_bins(ds.lat, bins=bins, labels=labels)
    return grouped.mean().where(grouped.count() >= min_count)


def make_axes(ax, variables):
    """Return one axis per variable: ax for the first, twin axes for the rest."""
    var_axes = {variables[0]: ax}
    for k, var in enumerate(variables[1:]):
        twin = ax.twinx()
        twin.set_ylim(0, 3000)
        if k > 0:  # offset further right axes so they don't overlap
            twin.spines["right"].set_position(("axes", 1 + 0.15 * k))
        var_axes[var] = twin
    for var, a in var_axes.items():
        c = plot_settings[var]["color"]
        a.tick_params(axis="y", colors=c)
        a.yaxis.label.set_color(c)
    return var_axes


def add_reference_line(ax, var, with_label=True):
    s = plot_settings[var]
    if "ref" in s:
        ax.axhline(
            s["ref"],
            color=s["color"],
            ls="--",
            lw=0.8,
            label=s.get("ref_label") if with_label else None,
        )


# --- flight days and binning -------------------------------------------------
start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
flight_days = sorted(
    (pd.Timestamp(info["date"]), flight_id)
    for flight_id, info in flights.items()
    if start <= pd.Timestamp(info["date"]) <= end
)

binned_flights = {}
for day, flight_id in flight_days:
    hamp_day = ds_hamp.sel(time=day.strftime("%Y-%m-%d"))
    if hamp_day.time.size == 0:
        continue
    binned_flights[flight_id] = bin_by_lat(
        hamp_day, variables, lat_bins, lat_bin_centers, min_count_flight
    )

hamp_period = ds_hamp.sel(time=slice(start_date, end_date))
binned_mean = bin_by_lat(
    hamp_period, variables, lat_bins, lat_bin_centers, min_count_mean
)

# --- figure 1: mean + all flights -------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
var_axes = make_axes(ax, variables)

handles = []
for var, a in var_axes.items():
    s = plot_settings[var]
    for binned_day in binned_flights.values():
        a.plot(binned_day.lat_bins, binned_day[var], color=s["light"], lw=1, zorder=1)
    (h,) = a.plot(
        binned_mean.lat_bins,
        binned_mean[var],
        color=s["color"],
        lw=2,
        zorder=2,
        label=f"{var} mean",
    )
    handles.append(h)

# %%

# --- figure 2: one panel per flight -----------------------------------------
n_flights = len(binned_flights)
ncols = 3
nrows = math.ceil(n_flights / ncols)

fig2, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(3.2 * ncols, 2.4 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
)

shared_twins = {}  # so the right-hand axes are also shared across panels
for idx, (ax, (flight_id, binned_day)) in enumerate(
    zip(axes.flat, binned_flights.items())
):
    var_axes = make_axes(ax, variables)
    col = idx % ncols
    last_in_row = col == ncols - 1 or idx == n_flights - 1

    for var, a in var_axes.items():
        s = plot_settings[var]
        if a is not ax:
            if var in shared_twins:
                a.sharey(shared_twins[var])
            else:
                shared_twins[var] = a
            if not last_in_row:
                a.tick_params(labelright=False)
            else:
                a.set_ylabel(s["label"])
        a.plot(binned_mean.lat_bins, binned_mean[var], color=s["light"], lw=2)
        a.plot(binned_day.lat_bins, binned_day[var], color=s["color"], lw=1.2)
        add_reference_line(a, var, with_label=False)
        a.axhline(60, color="k", lw=0.5, ls="--", zorder=0)

    ax.set_title(flight_id, fontsize=9)

for ax in axes.flat[n_flights:]:
    ax.set_axis_off()
for ax in axes[-1, :]:
    ax.set_xlabel("latitude / °N")
for ax in axes[:, 0]:
    ax.set_ylabel(
        plot_settings[variables[0]]["label"], color=plot_settings[variables[0]]["color"]
    )

fig2.suptitle(f"Individual flights (dark) vs. mean (light), {start_date} to {end_date}")
# %%
