# %%

import xarray as xr
from xarray.groupers import BinGrouper, UniqueGrouper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import percusion
from percusion import utils
from doldrumsVerticalMotion import circleUtils
import matplotlib.colors as colors

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%

PERIODS = {
    "East Atlantic": ("2024-08-11", "2024-09-05"),
    "West Atlantic": ("2024-09-07", "2024-09-28"),
}

bin_to_center = lambda bins: (bins[:-1] + bins[1:]) / 2

cwv_threshold = 48

# %%

HAMP_CID = "ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y"
ds_hamp = xr.open_dataset(HAMP_CID, engine="zarr")

hamp_orcestra = ds_hamp.sel(time=slice(utils.campaign_start, utils.campaign_end))
hamp_orcestra = hamp_orcestra.dropna(dim="time", subset=["IWV"])
iwv_hamp_orcestra = hamp_orcestra["IWV"]
iwv_hamp_orcestra = iwv_hamp_orcestra.assign_coords(
    lat=hamp_orcestra["lat"], day=hamp_orcestra.time.dt.floor("D")
)
# %%

ds_ds = xr.open_dataset(
    "ipfs://bafybeibgeeqs5uhmbqy4hz4v3pihrfisiklcetisxkf63d2r473szaprwi", engine="zarr"
)

ds_ds = ds_ds.swap_dims({"circle": "circle_id"})
ds_ds = ds_ds.dropna(dim="circle_id", subset=["iwv_mean"])

iwv_ds_orcestra = ds_ds["iwv_mean"]

iwv_circle_min = np.empty(len(ds_ds.circle_id))
iwv_circle_max = np.empty(len(ds_ds.circle_id))
iwv_circle_std = np.empty(len(ds_ds.circle_id))
iwv_circle_transition = np.empty(len(ds_ds.circle_id), dtype=bool)

for i, circle_id in enumerate(ds_ds.circle_id.values):
    sonde_ids = circleUtils.get_sonde_serial_ids(ds_ds, circle_id)

    iwv_circle_i = ds_ds["iwv"].where(
        ds_ds.vaisala_serial_id.isin(sonde_ids), drop=True
    )

    iwv_circle_min[i] = iwv_circle_i.min().values
    iwv_circle_max[i] = iwv_circle_i.max().values
    iwv_circle_std[i] = iwv_circle_i.std().values

    if iwv_circle_min[i] < cwv_threshold < iwv_circle_max[i]:
        iwv_circle_transition[i] = True
    else:
        iwv_circle_transition[i] = False


ds_ds["iwv_circle_min"] = (("circle_id"), iwv_circle_min)
ds_ds["iwv_circle_max"] = (("circle_id"), iwv_circle_max)
ds_ds["iwv_circle_std"] = (("circle_id"), iwv_circle_std)
ds_ds["iwv_circle_transition"] = (("circle_id"), iwv_circle_transition)


ds_iwv = ds_ds[
    [
        "iwv_circle_min",
        "iwv_circle_max",
        "iwv_mean",
        "iwv_circle_transition",
        "iwv_circle_std",
    ]
].copy()
ds_iwv["circle_time_pd"] = pd.DatetimeIndex(ds_ds["circle_time"])

# %%

# WALES water vapor
store = (
    "https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/wales/wales_wv.zarr"
)
ds_wales_wv = xr.open_dataset(store, engine="zarr")
ds_wales_wv = ds_wales_wv.sel(time=slice(utils.campaign_start, utils.campaign_end))
ds_wales_wv = ds_wales_wv.dropna(dim="time", how="all", subset=["wv"])
# %%
# CWV from number of water molucles per unit volumn

wv_flags = ds_wales_wv["wv_flags"]
below_aircraft = ds_wales_wv.altitude <= ds_wales_wv.flight_altitude

n_v = ds_wales_wv["wv"].where((wv_flags == 0) & below_aircraft)  # molecules/m^3

vertical_resolution_wales = (
    ds_wales_wv.altitude[1:].values - ds_wales_wv.altitude[:-1].values
)

if np.max(vertical_resolution_wales) - np.min(vertical_resolution_wales) > 1e-5:
    raise ValueError(
        "Vertical resolution of WALES water vapor data is not constant. Please check the data."
    )
else:
    vertical_resolution_wales = vertical_resolution_wales[0]

molar_mass_water = 18.01528e-3  # kg/mol
Avogadro_number = 6.02214076e23  # molecules/mol

cwv_wales = (
    (n_v * vertical_resolution_wales).sum(dim="altitude")
    * molar_mass_water
    / Avogadro_number
)  # kg/m^2

# %%

valid = xr.where(wv_flags == 0, 1, 0).where(below_aircraft)
wales_mask = valid.mean(
    "altitude", skipna=True
)  # fraction valid, of available data only

crit_data_fraction = 0.90  # fraction of valid data points
cwv_wales_filtered = xr.where(
    (wales_mask >= crit_data_fraction),
    cwv_wales,
    0,
)

print(
    f"Fraction of WALES data points with more than {crit_data_fraction*100:.0f}% valid data: "
    f"{np.sum(wales_mask >= crit_data_fraction)/len(wales_mask)*100:.2f}%"
)

# %%
# Plotting


bins = np.arange(0, 68, 1.0)

x_values = np.empty(len(ds_iwv["circle_time_pd"]))
x_values[0] = 0

x_ticks = []
x_ticks.append(0)
x_labels = []
x_labels.append(ds_iwv["circle_time_pd"][0].dt.strftime("%m-%d").values)

for i, t in enumerate(ds_iwv["circle_time_pd"][1:], start=1):
    if t.dt.dayofyear != ds_iwv["circle_time_pd"][i - 1].dt.dayofyear:
        x_values[i] = x_values[i - 1] + 2

        x_ticks.append(x_values[i])
        x_labels.append(t.dt.strftime("%m-%d").values)
    else:
        x_values[i] = x_values[i - 1] + 0.25


# %%

fig = plt.figure(figsize=(10, 6), layout="constrained")
subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[1, 0.8])

ax = subfig_top.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 1]})
ax_lat = subfig_bottom.subplots(1, 2, sharey=True, sharex=True)

col_ds, col_hamp = "C0", "#444865"  # "#FC9241"
marker = "o"

plt.sca(ax[0])

scatter_kwargs = {"color": col_ds, "clip_on": False}
hlines_kwargs = {"color": "k", "alpha": 0.5, "linewidth": 1, "linestyle": ":"}

num_circles = len(ds_iwv["circle_time_pd"])
num_transition_circles = 0
num_smaller_cwv_circles = 0
num_larger_cwv_circles = 0

for i in range(num_circles):

    iwv_min_circle_i = ds_iwv["iwv_circle_min"][i]
    iwv_max_circle_i = ds_iwv["iwv_circle_max"][i]
    transition = ds_iwv["iwv_circle_transition"][i]

    if transition:
        alpha = 1.0
        num_transition_circles += 1
    else:
        alpha = 0.35
        num_larger_cwv_circles += 1

    plt.vlines(
        x_values[i],
        ds_iwv["iwv_circle_min"][i],
        ds_iwv["iwv_circle_max"][i],
        alpha=alpha,
        linewidth=1,
        **scatter_kwargs,
    )
    plt.scatter(
        x_values[i],
        ds_iwv["iwv_mean"][i],
        s=15,
        marker=marker,
        alpha=alpha,
        **scatter_kwargs,
    )

print(
    f"Out of {len(ds_ds.circle_id)}: \n {num_transition_circles} circles are transition circles. \n"
)

plt.axhline(cwv_threshold, **hlines_kwargs)

plt.xticks(x_ticks, x_labels, rotation=45)

plt.xlabel("flight date / MM-DD")
plt.ylabel("CWV / mm")

plt.xlim(0, x_ticks[-1] + 0.5)

plt.vlines(
    x_ticks[np.where(np.array(x_labels) == "09-06")[0][0]],
    30,
    38,
    color="k",
    alpha=1,
    linewidth=0.75,
    linestyle="solid",
)

for day in ["08-25", "09-19"]:
    plt.text(
        x_ticks[np.where(np.array(x_labels) == day)[0][0]],
        32,
        "East Atlantic" if day == "08-25" else "West Atlantic",
        fontsize=9,
        va="bottom",
        ha="center",
    )


plt.sca(ax[1])


hist_kwargs = {
    "density": True,
    "orientation": "horizontal",
    "bins": bins,
    "clip_on": True,
}

for h_type in ["step"]:
    alpha = 1.0 if h_type == "step" else 0.3

    pdf_values_ds, bins_cwv_ds, _ = ax[1].hist(
        ds_ds.iwv.values,
        histtype=h_type,
        color=col_ds,
        alpha=alpha,
        label="dropsondes",
        **hist_kwargs,
    )

    pdf_values_hamp, bins_cwv_hamp, _ = ax[1].hist(
        iwv_hamp_orcestra.values,
        histtype=h_type,
        color=col_hamp,
        alpha=alpha,
        label="HAMP",
        **hist_kwargs,
    )

    pdf_values_wales, bins_cwv_wales, _ = ax[1].hist(
        cwv_wales_filtered.values,
        histtype=h_type,
        color="C2",
        alpha=alpha,
        label="WALES",
        **hist_kwargs,
    )


plt.axhline(cwv_threshold, **hlines_kwargs)
plt.ylim(30, bins[-1])
plt.xlim(0, 0.08)
plt.legend(frameon=False)

ax[1].set_xlabel("PDF")

#

sigma_deg = 0.2
dx = 0.2
sigma_bins = int(sigma_deg / dx)

bins_lat = np.arange(2, 18 + dx, dx)
bins_lat_centers = bin_to_center(bins_lat)

from matplotlib.colors import LinearSegmentedColormap

hex_green_blue = [
    "#40916c",
    "#52b788",
    "#74c69d",
    "#95d5b2",
    "#b7e4c7",
    "#89c2d9",
    "#61a5c2",
    "#468faf",
    "#2a6f97",
    "#013a63",
]

cmap_green_blue = LinearSegmentedColormap.from_list("green_blue", hex_green_blue)
cmap_green_blue_r = cmap_green_blue.reversed()  # dark blue -> green

n_days = 12
x = np.concatenate(
    [np.linspace(0, 0.4, n_days // 2), np.linspace(0.6, 1, n_days - n_days // 2)]
)
day_colors = cmap_green_blue(x)

# cmap = plt.get_cmap("viridis")
# day_colors = cmap(np.linspace(0, 0.85, n_days))


for a, (name, (start, end)) in zip(ax_lat, PERIODS.items()):
    iwv_period = iwv_hamp_orcestra.sel(time=slice(start, end))
    iwv_lat_mean = iwv_period.groupby_bins("lat", bins=bins_lat).mean()

    groupers = {
        "lat": BinGrouper(bins=bins_lat, labels=bins_lat_centers),
        "day": UniqueGrouper(),
    }
    iwv_lat_daily_mean = iwv_period.groupby(groupers).mean()

    for day in iwv_lat_daily_mean.day.values:

        val_smoothed = gaussian_filter1d(
            iwv_lat_daily_mean.sel(day=day).values,
            sigma=sigma_bins,
            axis=0,
            mode="nearest",
        )

        a.plot(
            bins_lat_centers,
            val_smoothed,
            color=day_colors[np.where(iwv_lat_daily_mean.day.values == day)[0][0]],
            alpha=0.9,
            linewidth=0.5,
            zorder=10,
        )

    val_mean_smoothed = gaussian_filter1d(
        iwv_lat_mean.values, sigma=sigma_bins, axis=0, mode="nearest"
    )

    a.plot(
        bins_lat_centers,
        val_mean_smoothed,
        color=col_hamp,
        linewidth=2,
        zorder=20,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    a.text(
        10,
        32,
        name,
        fontsize=9,
        va="bottom",
        ha="center",
    )

    a.axhline(cwv_threshold, **hlines_kwargs)
    a.set_xlabel("latitude / °N")
    a.set_xlim(2, 18)


a.set_ylim(30, bins[-1])

ax_lat[0].set_ylabel("CWV / mm")


all_axes = list(ax) + list(ax_lat)

for a in all_axes:
    a.spines["left"].set_position(("outward", 5))

for a, lbl, off in zip(all_axes, ["a)", "b)", "c)", "d)"], [-0.1, -0.2, -0.15, -0.15]):
    a.text(
        off,
        1.05,
        lbl,
        transform=a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )

sns.despine(fig=fig)

plt.savefig(
    f"{PROJECT_ROOT}/figures/transition_circles.pdf",
    bbox_inches="tight",
)


cwv_pdf_max_ds = bin_to_center(bins_cwv_ds)[np.argmax(pdf_values_ds)]
cwv_pdf_max_hamp = bin_to_center(bins_cwv_hamp)[np.argmax(pdf_values_hamp)]
cwv_pdf_max_wales = bin_to_center(bins_cwv_wales[10:])[
    np.argmax(pdf_values_wales[10:])
]  # exclude the first bins (0-10 mm) for WALES as these are the filtered out data points with insufficient valid data
print(f"Maximum PDF value for dropsondes: {cwv_pdf_max_ds}")
print(f"Maximum PDF value for HAMP: {cwv_pdf_max_hamp}")
print(f"Maximum PDF value for WALES: {cwv_pdf_max_wales}")

# %%
## Print what fraction of data points were sampled below given CWV threshold

cwv_frac = cwv_threshold

hamp_frac_below_crit_wvp = np.sum(
    iwv_hamp_orcestra.dropna(dim="time") < cwv_frac
) / len(iwv_hamp_orcestra.dropna(dim="time"))

ds_frac_below_crit_wvp = np.sum(ds_ds.iwv.dropna(dim="sonde") < cwv_frac) / len(
    ds_ds.iwv.dropna(dim="sonde")
)

print(
    f"Fraction of values collected below {cwv_frac} mm: {hamp_frac_below_crit_wvp:.2f} (HAMP), {ds_frac_below_crit_wvp:.2f} (dropsondes)"
)

# %%
## Use below to analyse how successful a given flight sampled the edges and the center of the ITCZ
# date = "09-28"
# circle_ids_on_date = np.arange(len(ds_ds.circle_id))[circle_dates_str == date]
# iwv_circle_min_on_date = iwv_circle_min[circle_ids_on_date]
# iwv_circle_max_on_date = iwv_circle_max[circle_ids_on_date]

# ds_on_date = ds_ds.isel(circle_id=circle_ids_on_date)

# fig, ax = utils.base_map(coastline_kwargs={"color": "k"})

# for i_circle, circle_id in enumerate(ds_on_date.circle_id.values):

#     ds_circle = ds_on_date.sel(circle_id=circle_id)

#     iwv_min_circle_i = iwv_circle_min_on_date[i_circle]
#     iwv_max_circle_i = iwv_circle_max_on_date[i_circle]

#     if iwv_min_circle_i < cwv_threshold < iwv_max_circle_i:
#         color = "deepskyblue"
#     elif iwv_max_circle_i < cwv_threshold:
#         color = "red"
#     else:
#         color = "navy"

#     ax.scatter(
#         ds_circle["circle_lon"].values,
#         ds_circle["circle_lat"].values,
#         color=color,
#         s=100,
#     )


# %%
