# %%

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import percusion
from percusion import utils
from doldrumsVerticalMotion import circleUtils
import matplotlib.colors as colors

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%

bin_to_center = lambda bins: (bins[:-1] + bins[1:]) / 2

# %%

hamp_path_v2 = PROJECT_ROOT / "data" / "HAMP_with_IWV_IWP_LWP_TLWP_v2.nc"
ds_hamp = xr.open_dataset(hamp_path_v2)
hamp_orcestra = ds_hamp.sel(time=slice(utils.campaign_start, utils.campaign_end))
hamp_orcestra = hamp_orcestra.dropna(dim="time", subset=["IWV"])
iwv_hamp_orcestra = hamp_orcestra["IWV"]
# %%

ds_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)
ds_ds = ds_ds.swap_dims({"circle": "circle_id"})
ds_ds = ds_ds.dropna(dim="circle_id", subset=["iwv_mean"])

iwv_ds_orcestra = ds_ds["iwv_mean"]
iwv_ds_orcestra_np = iwv_ds_orcestra.to_numpy()

# %%

iwv_circle_min = np.empty(len(ds_ds.circle_id))
iwv_circle_max = np.copy(iwv_circle_min)
iwv_circle_mean = np.copy(iwv_circle_min)

for i, circle_id in enumerate(ds_ds.circle_id.values):
    sonde_ids = circleUtils.get_sonde_serial_ids(ds_ds, circle_id)

    iwv_circle_i = ds_ds["iwv"].where(
        ds_ds.vaisala_serial_id.isin(sonde_ids), drop=True
    )

    iwv_circle_min[i] = iwv_circle_i.min().values
    iwv_circle_max[i] = iwv_circle_i.max().values
    iwv_circle_mean[i] = iwv_circle_i.mean().values


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


cwv_threshold = 48
bins = np.arange(0, 75, 1.0)

transition_circles = np.where(
    (iwv_circle_min < cwv_threshold) & (iwv_circle_max > cwv_threshold)
)[0]

smaller_cwv_circles = np.where(iwv_circle_max < cwv_threshold)[0]
larger_cwv_circles = np.where(iwv_circle_min > cwv_threshold)[0]

# %%

circle_dates_str = pd.to_datetime(ds_ds.circle_time.values).strftime("%m-%d")
circles_day_mapped, circle_unique_days = pd.factorize(circle_dates_str)

x = np.copy(iwv_circle_min)
x[0] = 0

xticks = [0]
xticks_labels = [circle_unique_days[0]]

for i_doy, doy in enumerate(circles_day_mapped[1:], start=1):

    if doy != circles_day_mapped[i_doy - 1]:
        x[i_doy] = circles_day_mapped[i_doy] * 2
        xticks.append(x[i_doy])
        xticks_labels.append(circle_unique_days[doy])

    else:
        x[i_doy] = x[i_doy - 1] + 0.25


# %%

circle_numbers = np.arange(len(iwv_circle_min))


fig, ax = plt.subplots(
    1, 2, figsize=(10, 4), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
)

col_ds, col_hamp = "C0", "C1"

plt.sca(ax[0])

scatter_kwargs = {"color": col_ds, "clip_on": False}
hlines_kwargs = {"color": "k", "alpha": 0.5, "linewidth": 1, "linestyle": ":"}

for i_m, mask in enumerate(
    [transition_circles, ~np.isin(circle_numbers, transition_circles)]
):

    alpha = 1.0 if i_m == 0 else 0.35
    marker = "o"  # if i_m == 0 else "x"

    plt.vlines(
        x[mask],
        iwv_circle_min[mask],
        iwv_circle_max[mask],
        alpha=alpha,
        linewidth=1,
        **scatter_kwargs,
    )

    plt.scatter(
        x[mask],
        iwv_circle_mean[mask],
        s=15,
        marker=marker,
        alpha=alpha,
        **scatter_kwargs,
    )


plt.axhline(cwv_threshold, **hlines_kwargs)

plt.xticks(xticks, xticks_labels, rotation=45)

plt.xlabel("flight date / MM-DD")
plt.ylabel("CWV / mm")

plt.tight_layout()
plt.xlim(0, x[-1] + 0.5)
plt.scatter(
    xticks[np.where(np.array(xticks_labels) == "09-06")[0][0]],
    bins[0],
    marker=10,
    color=hlines_kwargs["color"],
    alpha=hlines_kwargs["alpha"],
)


plt.sca(ax[1])


hist_kwargs = {
    "density": True,
    "orientation": "horizontal",
    "bins": bins,
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
        color="#709D9D",
        alpha=alpha,
        label="WALES",
        **hist_kwargs,
    )


plt.axhline(cwv_threshold, **hlines_kwargs)
plt.ylim(30, bins[-1])
plt.xlim(0, 0.08)
plt.legend()

ax[1].set_xlabel("PDF")

for a in ax:
    a.spines["left"].set_position(("outward", 5))

panel_labels = ["a)", "b)"]
offsets = [-0.1, -0.2]
for a, lbl in zip(ax, panel_labels):
    a.text(
        offsets[ax.tolist().index(a)],
        1.05,
        lbl,
        transform=a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )

sns.despine()

plt.savefig(
    f"{PROJECT_ROOT}/figures/transition_circles.pdf",
    bbox_inches="tight",
)

print(
    f"Out of {len(ds_ds.circle_id)}: \n {len(transition_circles)} circles are transition circles, \n"
    f" {len(smaller_cwv_circles)} circles have CWV entirely below {cwv_threshold} mm, \n"
    f" {len(larger_cwv_circles)} circles have CWV entirely above {cwv_threshold} mm."
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


# # %%

# %%
