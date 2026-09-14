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
    cloud_mask.groupby_bins(iwv_hamp_interpolated, bins=bins).mean().compute() * 100
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
# --------------------------------------------------------------------------
# Three-panel figure, 2 rows:
#   a) RH / cloud-occurrence-frequency contour plot — full width, top row
#   b) longwave radiation flux + IWP — bottom row, two-thirds width
#   c) vertical velocity profiles by CWV threshold — bottom row, one-third
#      width
# --------------------------------------------------------------------------

fig = plt.figure(figsize=(9, 8))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], wspace=0.75, hspace=0.25)

ax_cloud_mask = fig.add_subplot(gs[0, :3])  # 2/3 width
ax_vertical_velocity = fig.add_subplot(gs[0, 3:5])  # 1/3 width
ax_rad_flux = fig.add_subplot(gs[1, :3])  # 2/3 width


# ---------------------------------------------------------------------
# Panel a) CWV / cloud-occurrence-frequency contour plot
# ---------------------------------------------------------------------
pos = ax_cloud_mask.get_position()

wv_plot = wales_wv_binned_iwv.plot.contourf(
    y="altitude",
    ax=ax_cloud_mask,
    levels=np.arange(0, 61, 3),
    cmap="Blues",
    alpha=0.9,
    add_colorbar=False,
)

cax_wv_x0 = 0.11

cax_wv = fig.add_axes(
    [
        cax_wv_x0,
        pos.y1 + 0.02,  # bottom
        0.5 * pos.width,  # width
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
    levels=np.arange(0.05, 0.51, 0.025) * 100,
    y="altitude",
    ax=ax_cloud_mask,
    label=f"IWV bin {bin}",
    cmap=cmap,
    norm=colors.Normalize(vmin=0, vmax=30),
    add_colorbar=False,
)

cax_bloud_mask = fig.add_axes(
    [
        cax_wv_x0 + 0.5 * pos.width + 0.02,  # left
        pos.y1 + 0.02,  # bottom
        0.5 * pos.width,  # width
        0.02,  # height
    ]
)

cb = fig.colorbar(
    cloud_mask_plot,
    cax=cax_bloud_mask,
    orientation="horizontal",
    label=r"frequency of (Z > $10^{-5}$ mm$^{6}$ m$^{-3}$) / %",
    shrink=0.75,
)

cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_ticks_position("top")

ax_cloud_mask.set_xlim(bin_centers[0], bin_centers[-1])
ax_cloud_mask.set_ylim(ymin=250, ymax=13e3)
ax_cloud_mask.set_ylabel("height / m")
ax_cloud_mask.set_xlabel(" ")

# ---------------------------------------------------------------------
# Panel b) longwave radiation flux (left axis) and IWP (right, twin axis)
# — pulled off panel a) and given its own subplot
# ---------------------------------------------------------------------
color_ful = "teal"
ful_binned_iwv.plot(ax=ax_rad_flux, color=color_ful)
ax_rad_flux.set_ylabel("longwave radiation flux / W m$^{-2}$", color=color_ful)
ax_rad_flux.tick_params(axis="y", colors=color_ful)
ax_rad_flux.set_ylim(200, 300)
ax_rad_flux.spines["left"].set_color(color_ful)
ax_rad_flux.set_xlim(bin_centers[0], bin_centers[-1])
ax_rad_flux.set_xlabel("CWV / mm")
ax_rad_flux.set_title("")

ax1_twin = ax_rad_flux.twinx()
color_iwp = "#F2935C"
(hamp_orcestra_binned_iwv.IWP / 1e3).plot.line(ax=ax1_twin, color=color_iwp)
ax1_twin.set_ylabel("IWP / kg m$^{-2}$", color=color_iwp)
ax1_twin.tick_params(axis="y", colors=color_iwp)
ax1_twin.set_ylim(0, 0.7)
ax1_twin.spines["right"].set_color(color_iwp)
ax1_twin.spines["left"].set_color(color_ful)
ax1_twin.set_title("")

# ---------------------------------------------------------------------
# Panel c) vertical velocity profiles by CWV threshold
# ---------------------------------------------------------------------
cmap = plt.get_cmap("Blues")
norm = colors.Normalize(vmin=36, vmax=68)

tcwv_levels = [48]
tcwv_colors = [cmap(norm(level)) for level in tcwv_levels]

cols = ["#736A65", "C0"]

for i_bounds, bounds in enumerate([(0, 48), (48, 100)]):

    iwv_i = (
        ds_ds["wvel"].where(
            (ds_ds["iwv_mean"] >= bounds[0]) & (ds_ds["iwv_mean"] < bounds[1]),
            drop=True,
        )
        * 100
    )  # convert to cm/s

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
        ax=ax_vertical_velocity,
        label=label,
        color=cols[i_bounds],
    )

ax_vertical_velocity.legend(
    frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=1
)
ax_vertical_velocity.axvline(0, color="k", alpha=0.5, linestyle=":")
ax_vertical_velocity.set_xlim(-1, 1.5)
ax_vertical_velocity.set_ylim(250, 13e3)
ax_vertical_velocity.set_xlabel("vertical velocity / cm s$^{-1}$")
ax_vertical_velocity.set_ylabel(" ")

# ---------------------------------------------------------------------
# Panel labels a) - c)
# ---------------------------------------------------------------------
panel_labels = ["a)", "b)", "c)"]
for a, lbl in zip([ax_cloud_mask, ax_vertical_velocity, ax_rad_flux], panel_labels):
    offset = 0.00 if a == ax_cloud_mask else 0.05
    a.text(
        -0.1 - offset,
        1.1,
        lbl,
        transform=a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )

# ---------------------------------------------------------------------
# Spines
# ---------------------------------------------------------------------
sns.despine(ax=ax_cloud_mask, top=True, right=True)
sns.despine(ax=ax_rad_flux, top=True, right=True)
for a in [ax_rad_flux, ax1_twin]:
    sns.despine(ax=a, top=True, right=False)
sns.despine(ax=ax_vertical_velocity, top=True, right=True)

plt.savefig(
    f"{PROJECT_ROOT}/figures/itcz_cwv_space.pdf",
    bbox_inches="tight",
)

# %%

ds_sorted = ds_ds.sortby("iwv_mean")
ds_sorted.wvel.plot.pcolormesh(y="altitude", vmin=-0.075, vmax=0.075, cmap="bwr")

n = 5

current_ax = plt.gca()
tick_positions = np.arange(len(ds_sorted.circle_id))[::n]
tick_labels = [f"{val:.1f}" for val in ds_sorted["iwv_mean"].values[::n]]
current_ax.set_xticks(tick_positions)
current_ax.set_xticklabels(tick_labels, rotation=45)
current_ax.set_xlabel("CWV (mm)")

current_ax.set_ylim(ymin=250, ymax=13e3)
sns.despine()
# %%
