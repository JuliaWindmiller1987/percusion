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
from metpy.calc import density, mixing_ratio_from_specific_humidity, moist_static_energy
from metpy.units import units

from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %% --- helper functions ---------------------------------------------------------


def calc_circle_density(ds_ds):
    """Density (circle_id, altitude) from the circle-mean profiles."""
    r_mean = mixing_ratio_from_specific_humidity(ds_ds.q_mean * units("kg/kg"))
    rho = density(ds_ds.p_mean * units.Pa, ds_ds.ta_mean * units.K, r_mean)
    return rho


def calc_mse(ds_ds):
    """Moist static energy from the individual sonde profiles h = cp*T + g*z + Lv*q, in J kg-1."""
    h = moist_static_energy(
        ds_ds.altitude * units.m,
        ds_ds.ta * units.K,
        ds_ds.q * units("kg/kg"),
    )
    return h.metpy.convert_units("kJ/kg").metpy.dequantify()


def calc_streamfunction(ds_ds, smooth_sigma=None):
    rho_ref = ds_ds.rho.mean("iwv_mean")
    psi = (rho_ref * ds_ds.wvel).cumsum(dim="iwv_mean") / ds_ds.sizes["iwv_mean"]
    psi = psi * 1e3  # kg m^-2 s^-1 -> g m^-2 s^-1
    psi.attrs["long_name"] = "streamfunction"
    psi.attrs["units"] = "g m$^{-2}$ s$^{-1}$"

    if smooth_sigma is not None:
        psi = psi.copy(
            data=scipy.ndimage.gaussian_filter(
                psi.values, sigma=[smooth_sigma[d] for d in psi.dims]
            )
        )
    return psi


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

# Calculate streamfunction

ds_iwv_space = (
    ds_ds.sel(circle_id=ds_ds.circle_id.notnull())
    .dropna(dim="circle_id", subset=["iwv_mean"])
    .dropna(dim="sonde", subset=["iwv"])
    .sortby("iwv_mean", ascending=False)
    .sortby("iwv")
    .swap_dims(circle_id="iwv_mean")
    .swap_dims(sonde="iwv")
)

ds_iwv_space["rho"] = calc_circle_density(ds_iwv_space).metpy.dequantify()
psi = calc_streamfunction(ds_iwv_space, smooth_sigma={"iwv_mean": 3, "altitude": 10})


ds_iwv_space["mse"] = calc_mse(ds_iwv_space).metpy.dequantify()
mse_binned = ds_iwv_space["mse"].groupby_bins(ds_iwv_space["iwv"], bins=bins).mean()

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
# Three-panel figure, 3 rows (full width, different heights):
#   a) RH / cloud-occurrence-frequency contour plot
#   b) vertical velocity
#   c) longwave radiation flux + IWP
# --------------------------------------------------------------------------


def add_cbar(mappable, cax, title, frac=0.9):
    pos = cax.get_position()
    cax.set_position(
        [
            pos.x0,
            pos.y0 + 0.5 * (1 - frac) * pos.height,  # move up by half the reduction
            pos.width,
            frac * pos.height,
        ]
    )
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(title)
    return cb


fig = plt.figure(figsize=(10, 7.5))
gs = fig.add_gridspec(
    3,
    5,
    height_ratios=[2, 2, 1.5],
    width_ratios=[1, 0.01, 0.025, 0.075, 0.025],  # panels | colour bars
    hspace=0.075,
    wspace=0.04,
)

ax_cloud_mask = fig.add_subplot(gs[0, 0])
ax_vertical_velocity = fig.add_subplot(gs[1, 0], sharex=ax_cloud_mask)
ax_rad_flux = fig.add_subplot(gs[2, 0], sharex=ax_cloud_mask)

cax_wv = fig.add_subplot(gs[0, 2])
cax_cloud_mask = fig.add_subplot(gs[0, 4])
cax_mse = fig.add_subplot(gs[1, 2])

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

add_cbar(wv_plot, cax_wv, "RH / %")
add_cbar(
    cloud_mask_plot, cax_cloud_mask, "freq. of Z > $10^{-5}$ mm$^{6}$ m$^{-3}$ / %"
)

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
# Panel b) streamfunction (vertical velocity) contour plot and MSE color plot
# ---------------------------------------------------------------------

fig_psi = psi.plot.contour(
    x="iwv_mean",
    y="altitude",
    ax=ax_vertical_velocity,
    levels=np.arange(-1, 5.1, 1),
    colors="k",
    linewidths=1,
)

psi_plot = psi.sel(altitude=slice(250, 12e3))  # only the plotted range
imax = psi_plot.argmax(dim=["iwv_mean", "altitude"])
pmax = psi_plot.isel(imax)

ax_vertical_velocity.plot(pmax.iwv_mean, pmax.altitude, "k+", ms=6)
ax_vertical_velocity.annotate(
    f"{float(pmax):.1f}",
    (float(pmax.iwv_mean), float(pmax.altitude)),
    xytext=(4, 4),
    textcoords="offset points",
    fontsize=7,
)

mse_lim_max, mse_lim_min = 345, 330

mse_fig = mse_binned.plot.contourf(
    x="iwv_bins",
    y="altitude",
    ax=ax_vertical_velocity,
    levels=np.arange(320, 350, 1),
    cmap="BrBG",
    vmin=mse_lim_min,
    vmax=mse_lim_max,
    add_colorbar=False,
)

pos_w = ax_vertical_velocity.get_position()

add_cbar(mse_fig, cax_mse, "MSE / kJ kg$^{-1}$")

ax_vertical_velocity.set_xlabel("")
ax_vertical_velocity.set_ylabel("height / m")
ax_vertical_velocity.set_ylim(ymin=250, ymax=12e3)


# ---------------------------------------------------------------------
# Panel c) longwave radiation flux (left axis) and IWP (right, twin axis)
# ---------------------------------------------------------------------
color_ful = "C1"
ful_binned_iwv.plot(ax=ax_rad_flux, color=color_ful)
ful_bacardi_low_lat_binned_iwv.plot(ax=ax_rad_flux, color=color_ful, linestyle="--")
ful_bacardi_high_lat_binned_iwv.plot(ax=ax_rad_flux, color=color_ful, linestyle=":")

ax_rad_flux.set_ylabel(" ULRF / W m$^{-2}$", color=color_ful)
ax_rad_flux.tick_params(axis="y", colors=color_ful)
ax_rad_flux.set_ylim(200, 300)
ax_rad_flux.spines["left"].set_color(color_ful)
ax_rad_flux.set_xlim(bin_centers[0], bin_centers[-1])
ax_rad_flux.set_xlabel("CWV / mm")
ax_rad_flux.set_title("")

ax1_twin = ax_rad_flux.twinx()
color_iwp = "C0"
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
# Panel labels a) - c)
# ---------------------------------------------------------------------
panel_labels = ["a)", "b)", "c)"]
for a, lbl in zip([ax_cloud_mask, ax_vertical_velocity, ax_rad_flux], panel_labels):
    a.text(
        -0.15,
        1.0,
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

for a in [ax_cloud_mask, ax_vertical_velocity]:
    a.tick_params(axis="x", labelbottom=False)
    a.set_xlabel("")

plt.savefig(
    f"{PROJECT_ROOT}/figures/itcz_cwv_space.pdf",
    bbox_inches="tight",
)

# %%
