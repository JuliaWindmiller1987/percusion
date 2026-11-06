# %%

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import percusion
from percusion import utils
from doldrumsVerticalMotion import circleUtils
import matplotlib.colors as colors
import pandas as pd
import scipy.ndimage

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%
# HAMP (passive)

ds_hamp = xr.open_dataset(
    "ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y", engine="zarr"
)
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
# specMACS stereo heights

specmacs_stereo_heights = xr.open_dataset(
    PROJECT_ROOT / "data" / "stereo_height_vs_CWV_hist.nc"
)

specmacs_stereo_heights["frequency"] = (
    100 * specmacs_stereo_heights.histogram / specmacs_stereo_heights.ntot
)
specmacs_stereo_heights["frequency"].attrs["units"] = "%"

iwv_bins = pd.IntervalIndex(pd.cut(specmacs_stereo_heights.cwv.values, bins=bins))
specmacs_stereo_heights = specmacs_stereo_heights.assign_coords(
    IWV_bins=(specmacs_stereo_heights.cwv.dims, iwv_bins)
)
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

lat_threshold = 12

hamp_orcestra_low_lat = hamp_orcestra.where(
    hamp_orcestra.lat <= lat_threshold, drop=True
)
hamp_orcestra_high_lat = hamp_orcestra.where(
    hamp_orcestra.lat > lat_threshold, drop=True
)

hamp_orcestra_low_lat_binned_iwv = hamp_orcestra_low_lat.groupby_bins(
    hamp_orcestra_low_lat.IWV, bins=bins
).mean()

hamp_orcestra_high_lat_binned_iwv = hamp_orcestra_high_lat.groupby_bins(
    hamp_orcestra_high_lat.IWV, bins=bins
).mean()

# %%
# Upwelling longwave radiation binned by IWV
ds_bacardi = xr.open_dataset(
    "ipfs://bafybeiaoalflfftmsfqakenwp5gpxnxfjhaxkrjq7gpa4ucpbwj5jdb6qi", engine="zarr"
)

ful_bacardi = ds_bacardi["FUL"]
hamp_bacardi_interpolated = hamp_orcestra[["IWV", "lat"]].interp(time=ful_bacardi.TIME)
ful_binned_iwv = ful_bacardi.groupby_bins(
    hamp_bacardi_interpolated.IWV, bins=bins
).mean()

ful_bacardi_low_lat = ful_bacardi.where(
    hamp_bacardi_interpolated.lat <= lat_threshold, drop=True
)
ful_bacardi_high_lat = ful_bacardi.where(
    hamp_bacardi_interpolated.lat > lat_threshold, drop=True
)

ful_bacardi_low_lat_binned_iwv = ful_bacardi_low_lat.groupby_bins(
    hamp_bacardi_interpolated.IWV.where(
        hamp_bacardi_interpolated.lat <= lat_threshold, drop=True
    ),
    bins=bins,
).mean()

ful_bacardi_high_lat_binned_iwv = ful_bacardi_high_lat.groupby_bins(
    hamp_bacardi_interpolated.IWV.where(
        hamp_bacardi_interpolated.lat > lat_threshold, drop=True
    ),
    bins=bins,
).mean()

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

fig = plt.figure(figsize=(9, 6))
gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], wspace=0.75, hspace=0.25)

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
    label=r"freq. of Z > $10^{-5}$ mm$^{6}$ m$^{-3}$ / %",
    shrink=0.75,
)


cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_ticks_position("top")

specmacs_stereo_heights.frequency.plot.contour(
    levels=[0.01, 0.02, 0.04],
    y="height",
    ax=ax_cloud_mask,
    colors="k",
    linewidths=1,
    linestyles=[":", "--", "-"],
)

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
ful_bacardi_low_lat_binned_iwv.plot(ax=ax_rad_flux, color=color_ful, linestyle="--")
ful_bacardi_high_lat_binned_iwv.plot(ax=ax_rad_flux, color=color_ful, linestyle=":")

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
(hamp_orcestra_low_lat_binned_iwv.IWP / 1e3).plot.line(
    ax=ax1_twin, color=color_iwp, linestyle="--"
)
(hamp_orcestra_high_lat_binned_iwv.IWP / 1e3).plot.line(
    ax=ax1_twin, color=color_iwp, linestyle=":"
)


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

cwv_threshold = 48  # mm
tcwv_levels = [cwv_threshold]
tcwv_colors = [cmap(norm(level)) for level in tcwv_levels]

cols = ["#736A65", "C0"]
line_styles = ["-", "--", ":"]

for i_bounds, bounds in enumerate([(0, cwv_threshold), (cwv_threshold, 100)]):

    if bounds == (cwv_threshold, 100):
        lat_bounds_array = [(-10, 30), (-10, 12), (12, 30)]
    else:
        lat_bounds_array = [(-10, 30)]

    for i_lat, lat_bounds in enumerate(lat_bounds_array):
        iwv_i = (
            ds_ds["wvel"].where(
                (ds_ds["iwv_mean"] >= bounds[0])
                & (ds_ds["iwv_mean"] < bounds[1])
                & (ds_ds["circle_lat"] >= lat_bounds[0])
                & (ds_ds["circle_lat"] < lat_bounds[1]),
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
            label = (
                f"{bounds[0]} mm $\leq$ CWV < {bounds[1]} mm ({num_circles_i} circles)"
            )

        iwv_mean_i = iwv_i.mean("circle_id")
        iwv_mean_i = iwv_mean_i.sel(altitude=slice(0, 12.5e3))

        iwv_mean_i.plot.line(
            y="altitude",
            ax=ax_vertical_velocity,
            label=label,
            color=cols[i_bounds],
            linestyle=line_styles[i_lat],
        )

ax_vertical_velocity.legend(
    frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1
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
    offset = 0.00  # if a == ax_cloud_mask else 0.05
    a.text(
        -0.1 - offset,
        1.2,
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

from metpy.calc import density, mixing_ratio_from_specific_humidity, moist_static_energy
from metpy.units import units


def calc_circle_density(ds_ds):
    """Density (circle_id, altitude) from the circle-mean profiles."""
    r_mean = mixing_ratio_from_specific_humidity(ds_ds.q_mean * units("kg/kg"))
    rho = density(ds_ds.p_mean * units.Pa, ds_ds.ta_mean * units.K, r_mean)
    return rho


def calc_mse(ds_ds):
    """Moist static energy h = cp*T + g*z + Lv*q, in J kg-1."""
    h = moist_static_energy(
        ds_ds.altitude * units.m,
        ds_ds.ta * units.K,
        ds_ds.q * units("kg/kg"),
    )
    return h.metpy.convert_units("kJ/kg").metpy.dequantify()


# Calculate streamfunction

# %%

ds_iwv_space = (
    ds_ds.sel(circle_id=ds_ds.circle_id.notnull())  # drop circles without an ID
    .dropna(dim="circle_id", subset=["iwv_mean"])  # drop circles without CWV
    .dropna(dim="sonde", subset=["iwv"])
    .sortby("iwv_mean", ascending=False)  # moist -> dry
    .sortby("iwv")
    .swap_dims(circle_id="iwv_mean")  # iwv_mean becomes the dimension
    .swap_dims(sonde="iwv")
)

ds_iwv_space["rho"] = calc_circle_density(ds_iwv_space).metpy.dequantify()
ds_iwv_space["mse"] = calc_mse(ds_iwv_space).metpy.dequantify()

# %%

n_missing = ds_iwv_space.wvel.isnull().sum("iwv_mean")
n_missing.plot(y="altitude")
plt.xlabel("number of missing vertical velocity measurements")
plt.xlim(xmin=0)
# %%

campaign_periods = {
    "all": ("2024-08-11", "2024-09-29"),
    "east (Sal)": ("2024-08-11", "2024-09-06"),
    "west (Barbados)": ("2024-09-07", "2024-09-29"),
}

fig, axes = plt.subplots(
    len(campaign_periods),
    1,
    figsize=(6, 2.5 * len(campaign_periods)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for i, (period, (start, end)) in enumerate(campaign_periods.items()):

    plt.sca(axes[i])

    t_circle = ds_iwv_space.circle_time
    t_sonde = ds_iwv_space.launch_time
    t0 = np.datetime64(start)
    t1 = np.datetime64(end) + np.timedelta64(1, "D")

    ds_iwv_space_period = ds_iwv_space.sel(
        iwv_mean=(t_circle >= t0) & (t_circle < t1),
        iwv=(t_sonde >= t0) & (t_sonde < t1),
    )

    rho_ref = ds_iwv_space_period.rho.mean("iwv_mean")
    psi = (rho_ref * ds_iwv_space_period.wvel).cumsum(
        dim="iwv_mean"
    ) / ds_iwv_space_period.sizes["iwv_mean"]

    psi.attrs["long_name"] = "streamfunction"
    psi.attrs["units"] = "kg m$^{-2}$ s$^{-1}$"

    sigma = {"iwv_mean": 3, "altitude": 10}  # iwv_mean corresponds to number of circles
    psi_filtered = psi.copy(
        data=scipy.ndimage.gaussian_filter(
            psi.values, sigma=[sigma[d] for d in psi.dims]
        )
    )

    psi_lim = 0.008  # kg m^-2 s^-1

    # psi_lim = np.abs(psi_filtered).quantile(0.99).values
    # print(f"period {period}: psi_lim = {psi_lim:.3f} kg m^-2 s^-1")

    psi_filtered.plot.contour(
        x="iwv_mean",
        y="altitude",
        levels=20,
        vmin=-psi_lim,
        vmax=psi_lim,
        colors="k",
        linewidths=1,
    )

    mse_binned = (
        ds_iwv_space_period["mse"]
        .groupby_bins(ds_iwv_space_period["iwv"], bins=bins)
        .mean()
    )

    # mse_binned = mse_binned - ds_iwv_space_period["mse"].mean("iwv")
    # mse_lim_max = np.abs(mse_binned).quantile(0.95).values
    # print(f"period {period}: mse_lim = {mse_lim_max:.1f} J kg^-1")
    # mse_lim_max = 4000
    # mse_lim_min = -1* mse_lim_max

    # mse_lim_max = np.abs(mse_binned).quantile(0.8).values
    # mse_lim_min = np.abs(mse_binned).quantile(0.05).values
    # print(f"period {period}: mse_lim = {mse_lim_min:.1f} to {mse_lim_max:.1f} kJ kg^-1")

    mse_lim_max, mse_lim_min = 345, 330

    mse_binned.plot.contourf(
        x="iwv_bins",
        y="altitude",
        levels=20,
        cmap="RdBu_r",
        vmin=mse_lim_min,
        vmax=mse_lim_max,
    )

    plt.title(period)

for ax in axes[:-1]:
    ax.set_xlabel("")

plt.ylim(ymin=250, ymax=12e3)
plt.xlim(bin_centers[0], bin_centers[-1])
# %%
