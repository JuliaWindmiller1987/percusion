# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import percusion
from percusion import utils
from doldrumsVerticalMotion import circleUtils
import matplotlib.colors as colors

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%
# HAMP (passive)
hamp_path_v2 = PROJECT_ROOT / "data" / "HAMP_with_IWV_IWP_LWP_TLWP_v2.nc"
ds_hamp = xr.open_dataset(hamp_path_v2)
hamp_orcestra = ds_hamp.sel(time=slice(utils.campaign_start, utils.campaign_end))

iwv_hamp_orcestra = hamp_orcestra["IWV"]
cwv_xmin, cwv_xmax = float(iwv_hamp_orcestra.quantile(0.1).values), float(
    iwv_hamp_orcestra.quantile(0.9).values
)

cwv_xmin, cwv_xmax = np.round([cwv_xmin, cwv_xmax], 0)

# %%

bins = np.arange(cwv_xmin, cwv_xmax + 1, 1.0)
bin_centers = (bins[:-1] + bins[1:]) / 2

# %%
# Dropsondes
ds_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)
ds_ds = ds_ds.swap_dims({"circle": "circle_id"})


# %%
# HAMP (active)
ds_hamp_active = xr.open_dataset(
    "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a", engine="zarr"
)

ds_hamp_active = ds_hamp_active.sel(
    time=slice(utils.campaign_start, utils.campaign_end)
)

# %%
cloud_mask = xr.where(ds_hamp_active["radar_reflectivity"] > 1e-5, 1, 0)


# %%
# WALES water vapor
store = (
    "https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/wales/wales_wv.zarr"
)
ds_wales_wv = xr.open_dataset(store, engine="zarr")
ds_wales_wv = ds_wales_wv.sel(time=slice(utils.campaign_start, utils.campaign_end))

k_B = 1.380649e-23  # J/K

T = ds_wales_wv["airtemperature"]  # K
n_v = ds_wales_wv["wv"]  # molecules/m^3

e = n_v * k_B * T  # Pa

es = 611.2 * np.exp(
    17.67 * (T - 273.15) / (T - 29.65)
)  # Bolton formula for saturation vapor pressure over liquid water, in Pa

ds_wales_wv["RH"] = 100 * e / es


# %%
# Cloud mask binned by IWV

iwv_hamp_interpolated = iwv_hamp_orcestra.interp(time=ds_hamp_active.time)
cloud_mask_binned_iwv = (
    cloud_mask.groupby_bins(iwv_hamp_interpolated, bins=bins).mean().compute()
)

# %%
# HAMP passive binned by IWV

hamp_orcestra_binned_iwv = hamp_orcestra.groupby_bins(
    iwv_hamp_orcestra, bins=bins
).mean()

# %%
# Upwelling longwave radiation binned by IWV
ds_bacardi = xr.open_dataset(
    "ipfs://bafybeiaoalflfftmsfqakenwp5gpxnxfjhaxkrjq7gpa4ucpbwj5jdb6qi", engine="zarr"
)

ful_bacardi = ds_bacardi["FUL"]
iwv_bacardi_interpolated = iwv_hamp_orcestra.interp(time=ful_bacardi.TIME)
ful_binned_iwv = ful_bacardi.groupby_bins(iwv_bacardi_interpolated, bins=bins).mean()

# %%
# WALES water vapor binned by IWV

iwv_hamp_interpolated_wales = iwv_hamp_orcestra.interp(time=ds_wales_wv.time)

wales_wv_binned_iwv = (
    ds_wales_wv["RH"]
    .groupby_bins(iwv_hamp_interpolated_wales, bins=bins)
    .mean()
    .compute()
)

# %%
fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 4),
    sharey=True,
    gridspec_kw={"width_ratios": [2, 1], "wspace": 0.4},
)


pos = ax[0].get_position()

wv_plot = wales_wv_binned_iwv.plot.contourf(
    y="altitude",
    ax=ax[0],
    levels=np.arange(0, 61, 3),
    cmap="Blues",
    alpha=0.9,
    add_colorbar=False,
)

cax_wv = fig.add_axes(
    [
        pos.x0 + 0.1 * pos.width,  # left
        pos.y1 + 0.22,  # bottom
        0.8 * pos.width,  # width
        0.02,  # height
    ]
)

cb = fig.colorbar(
    wv_plot,
    cax=cax_wv,
    orientation="horizontal",
    label=r"RH / %",
    shrink=0.75,
)

cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_ticks_position("top")

cmap = plt.get_cmap("pink_r").copy()
cmap.set_under((1, 1, 1, 0))  # transparent

cloud_mask_plot = cloud_mask_binned_iwv.plot.contourf(
    levels=np.arange(0.05, 0.51, 0.025),
    y="altitude",
    ax=ax[0],
    label=f"IWV bin {bin}",
    cmap=cmap,
    norm=colors.Normalize(vmin=0, vmax=0.3),
    add_colorbar=False,
)


cax_cloud_mask = fig.add_axes(
    [
        pos.x0 + 0.1 * pos.width,  # left
        pos.y1 + 0.05,  # bottom
        0.8 * pos.width,  # width
        0.02,  # height
    ]
)

cb = fig.colorbar(
    cloud_mask_plot,
    cax=cax_cloud_mask,
    orientation="horizontal",
    label=r"frequency of (Z > $10^{-5}$ mm$^{6}$ m$^{-3}$) / %",
    shrink=0.75,
)

cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_ticks_position("top")


ax0_twin = ax[0].twinx()
color_ful = "teal"
ful_binned_iwv.plot(ax=ax0_twin, color=color_ful)
ax0_twin.set_ylabel("longwave radiation flux / W m$^{-2}$", color=color_ful)
ax0_twin.tick_params(axis="y", colors=color_ful)
ax0_twin.set_ylim(200, 300)
ax0_twin.spines["right"].set_color(color_ful)

ax0_twin2 = ax[0].twinx()
color_iwp = "#F2935C"  #
ax0_twin2.spines["right"].set_position(("outward", 52))
ax0_twin2.spines["right"].set_color(color_iwp)
ax0_twin2.tick_params(axis="y", colors=color_iwp)
(hamp_orcestra_binned_iwv.IWP / 1e3).plot.line(ax=ax0_twin2, color=color_iwp)
ax0_twin2.set_ylabel("IWP / kg m$^{-2}$", color=color_iwp)
ax0_twin2.set_ylim(0, 0.7)


ax[0].set_xlim(bin_centers[0], bin_centers[-1])
ax[0].set_ylim(ymin=250, ymax=13e3)
ax[0].set_ylabel("height / m")
ax[0].set_xlabel(" CWV / mm")


cmap = plt.get_cmap("Blues")
norm = colors.Normalize(vmin=36, vmax=68)

tcwv_levels = [48]
tcwv_colors = [cmap(norm(level)) for level in tcwv_levels]

cols = ["#736A65", "C0"]

for i_bounds, bounds in enumerate([(0, 48), (48, 100)]):

    iwv_i = ds_ds["wvel"].where(
        (ds_ds["iwv_mean"] >= bounds[0]) & (ds_ds["iwv_mean"] < bounds[1]), drop=True
    )

    num_circles_i = len(iwv_i.circle_id)

    if bounds[0] == 0:
        label = f"CWV < {bounds[1]} mm ({num_circles_i} circles)"

    elif bounds[1] == 100:
        label = f"CWV $\geq$ {bounds[0]} mm ({num_circles_i} circles)"

    else:
        label = f"{bounds[0]} mm $\leq$ CWV < {bounds[1]} mm ({num_circles_i} circles)"

    iwv_mean_i = iwv_i.mean("circle_id")
    iwv_mean_i = iwv_mean_i.sel(altitude=slice(0, 12.5e3))

    iwv_mean_i.plot.line(
        y="altitude",
        ax=ax[1],
        label=label,
        color=cols[i_bounds],
    )

ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=1)
ax[1].axvline(0, color="k", alpha=0.5, linestyle=":")
ax[1].set_xlim(-0.01, 0.015)
# ax[1].set_ylim(0, 12.5e3)
ax[1].set_xlabel("vertical velocity / m s$^{-1}$")
ax[1].set_ylabel(" ")

for a in [ax[0], ax0_twin, ax0_twin2]:
    sns.despine(ax=a, top=True, right=False)
sns.despine(ax=ax[1], top=True, right=True)

plt.savefig(
    f"{PROJECT_ROOT}/figures/itcz_cwv_space.pdf",
    bbox_inches="tight",
)

# %%
