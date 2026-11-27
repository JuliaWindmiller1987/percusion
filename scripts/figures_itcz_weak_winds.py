# %%

from orcestra import get_flight_segments
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm

import easygems.healpix as egh
import seaborn as sns
import cartopy.crs as ccrs

import importlib
import percusion
from pathlib import Path

importlib.reload(percusion)
from percusion.utils import base_map, get_halo_position
from percusion.datasets import open_dataset

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]


# %%
flights = get_flight_segments()["HALO"]
flight_id = "HALO-20240907a"
flight = flights[flight_id]

flight_start, flight_end = flight["takeoff"], flight["landing"]
flight_date = flight["date"]

segment_for_plot = [s for s in flight["segments"] if s["name"] == "circle_mid"][0]
segment_start = np.datetime64(segment_for_plot["start"])
segment_end = np.datetime64(segment_for_plot["end"])

# %%
# BAHAMAS dataset
tracks = get_halo_position().sel(time=flight_date)

# %%
# Dropsonde data
sondes_ds = open_dataset("dropsondes").swap_dims({"circle": "circle_id"})
sondes_ds = sondes_ds.assign_coords(altitude=sondes_ds.altitude / 1e3)
circles_flight_day = sondes_ds.sel(circle_time=flight_date)

wsp_crit = 3.0  # m/s
sfc_layer = slice(0, 0.1)  # km (lowest 100 m)
wsp_sfc = sondes_ds.wspd.sel(altitude=sfc_layer).mean("altitude")

ds_low_wsp = sondes_ds.where(wsp_sfc < wsp_crit, drop=True)
doldrum_sondes_id = ds_low_wsp.vaisala_serial_id

dol_mask = sondes_ds.vaisala_serial_id.isin(doldrum_sondes_id.values)
dol_sondes = sondes_ds.where(dol_mask, drop=True)

try:
    sondes_flight_day = sondes_ds.sel(launch_time=flight_date)
    print(
        f"Number of sondes launched on {flight_date}: {len(sondes_flight_day.launch_time)}"
    )
except KeyError:
    print("No sondes launched on this day.")

try:
    dol_sondes_flight_day = dol_sondes.sel(launch_time=flight_date)
    print(
        f"Number of doldrums sondes launched on {flight_date}: {len(dol_sondes_flight_day.launch_time)}"
    )
except KeyError:
    print("No doldrums sondes on this day.")

# %%
# WALES dataset
store = (
    "https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/wales/wales_wv.zarr"
)
ds_wales_wv = open_dataset("wales_wv").sel(time=slice(flight_start, flight_end))
ds_wales_wv = ds_wales_wv.where(ds_wales_wv.wv_flags == 0)

k_B = 1.380649e-23  # J/K

T = ds_wales_wv["airtemperature"]  # K
n_v = ds_wales_wv["wv"]  # molecules/m^3

e = n_v * k_B * T  # Pa

es = 611.2 * np.exp(
    17.67 * (T - 273.15) / (T - 29.65)
)  # Bolton formula for saturation vapor pressure over liquid water, in Pa

ds_wales_wv["RH"] = 100 * e / es

# %%
# HAMP dataset
ds_hamp_radar_flight = open_dataset("hamp_radar").sel(
    time=slice(flight_start, flight_end)
)

# %%
# LAM dataset
time = f"{flight_date} 16:00"

lam_ds = open_dataset("lam")[["uas", "vas"]].sel(time=time)
lam_ds = lam_ds.assign(sfcwind=lambda dx: np.hypot(dx.uas, dx.vas))
ds_lam_sel = lam_ds["sfcwind"].compute()

# %%

scatter_kwargs = {
    "s": 25,
    "edgecolor": "k",
    "linewidths": 1.0,
}


fig = plt.figure(figsize=(12, 8))

gs = fig.add_gridspec(
    2, 2, height_ratios=[1.5, 1], width_ratios=[4, 1], hspace=0.15, wspace=0.4
)

ax_lam = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_ds = fig.add_subplot(gs[0, 1])

ax_wales = fig.add_subplot(gs[1, :])

base_map(coastline_kwargs={"color": "k"}, ax=ax_lam)

vmin, vmax, vcenter = 0, 12, 3
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = plt.cm.BrBG.copy()
im = egh.healpix_show(ds_lam_sel, cmap=cmap, norm=norm, ax=ax_lam, alpha=1.0)


ax_lam.apply_aspect()
pos = ax_lam.get_position()

pos_ds = ax_ds.get_position()
ax_ds.set_position([pos_ds.x0, pos.y0, pos_ds.width, pos.height])

cbar_width = 0.01
cax_lam = fig.add_axes([pos.x1 + 0.015, pos.y0, cbar_width, pos.height])

cb_lam = fig.colorbar(
    im,
    cax=cax_lam,
    orientation="vertical",
    label=r"$u_{\text{sfc}}$ / m s$^{-1}$",
)

mask = (tracks.time >= segment_start) & (tracks.time <= segment_end)

ax_lam.plot(
    tracks.lon.where(~mask),
    tracks.lat.where(~mask),
    color="k",
    alpha=0.5,
)

ax_lam.plot(
    tracks.lon.where(mask),
    tracks.lat.where(mask),
    color="k",
)


ax_lam.scatter(
    sondes_flight_day.launch_lon,
    sondes_flight_day.launch_lat,
    facecolors="none",
    label=r"$\overline{u}_{0\text{--}100\,\mathrm{m}} \geq 3$ m s$^{-1}$",
    alpha=0.5,
    **scatter_kwargs,
)

ax_lam.scatter(
    dol_sondes_flight_day.launch_lon,
    dol_sondes_flight_day.launch_lat,
    color=None,
    label=r"$\overline{u}_{0\text{--}100\,\mathrm{m}} < 3$ m s$^{-1}$",
    alpha=0.75,
    zorder=10,
    facecolors="k",
    **scatter_kwargs,
)

ax_lam.legend(
    loc="upper left",
    ncol=1,
    labelcolor="k",
    fontsize=9,
)

ax_lam.set_xlabel("longitude / °")
ax_lam.set_ylabel("latitude / °")
ax_lam.spines[["right", "top"]].set_visible(False)
ax_lam.set_title(flight_id)
# ax_lam.legend(
#     loc="upper left",
#     ncol=1,
#     frameon=True,
#     edgecolor="k",
#     labelcolor="k",
#     fontsize=9,
#     facecolor="white",
#     framealpha=0.5,
# )

cmap_wales = plt.get_cmap("Blues").copy()
cmap_wales.set_under((1, 1, 1, 0))  # transparent

ds_wales_sel_segment = ds_wales_wv.sel(time=slice(segment_start, segment_end))
ds_wales_sel_segment = ds_wales_sel_segment.assign_coords(
    altitude=ds_wales_sel_segment.altitude / 1e3
)
im_wales = ds_wales_sel_segment["RH"].plot(
    alpha=0.9,
    x="time",
    levels=np.arange(0, 81, 3),
    ax=ax_wales,
    cmap=cmap_wales,
    add_colorbar=False,
)

# HAMP radar reflectivity

cmap = plt.get_cmap("pink_r").copy()
cmap.set_under((1, 1, 1, 0))  # transparent

reflectivity_segment = ds_hamp_radar_flight.radar_reflectivity.sel(
    time=slice(segment_start, segment_end)
)
reflectivity_segment = reflectivity_segment.assign_coords(
    altitude=reflectivity_segment.altitude / 1e3
)

im_hamp = reflectivity_segment.plot(
    y="altitude",
    ax=ax_wales,
    label=f"IWV bin {bin}",
    cmap=cmap,
    norm=colors.LogNorm(vmin=1e-5, vmax=1e5),
    add_colorbar=False,
)


pos = ax_wales.get_position()
ax_wales.set_position([pos.x0, pos.y0, pos.width * 0.82, pos.height])
pos = ax_wales.get_position()

cbar_width = 0.01
cax_wv = fig.add_axes([pos.x1 + 0.015, pos.y0, cbar_width, pos.height])
cax_wales = fig.add_axes([pos.x1 + 0.095, pos.y0, cbar_width, pos.height])

cb = fig.colorbar(
    im_hamp,
    cax=cax_wv,
    orientation="vertical",
    label=r"$Z_e$ / mm$^6$ m$^{-3}$",
)

cb_wales = fig.colorbar(
    im_wales,
    cax=cax_wales,
    orientation="vertical",
    label=r"RH / %",
)


circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).wvel.plot(
    y="altitude", ax=ax_ds, label="Vertical velocity", color="k"
)

circle_in_segment = circles_flight_day.sel(circle_id=segment_for_plot["segment_id"])

cwv_circle_in_segment = circle_in_segment.iwv_mean.values
wspd_100m_in_segment = circle_in_segment.wspd_mean.sel(altitude=sfc_layer).mean()


print(f"Mean IWV in circle segment: {cwv_circle_in_segment:.2f} kg/m^2")
print(f"Mean wind speed in circle segment: {wspd_100m_in_segment:.2f} m/s")
print(
    f"Min wind speed in circle segment: {circle_in_segment.wspd_mean.sel(altitude=sfc_layer).min().values:.2f} m/s"
)
print(
    f"Max wind speed in circle segment: {circle_in_segment.wspd_mean.sel(altitude=sfc_layer).max().values:.2f} m/s"
)

ax_ds.axvline(0, color="k", linestyle=":", linewidth=1, zorder=0)

ax_ds.set_xlabel("vertical velocity / m s$^{-1}$")
ax_ds.set_ylabel("height / km")
ax_ds.set_title(" ")
ax_ds.set_ylim(ymin=0, ymax=13)
ax_ds.set_xlim(xmin=-0.075, xmax=0.075)

ax_wales.set_ylim(ymin=0.5, ymax=13)

specMACS_start_times = [
    np.datetime64("2024-09-07T16:06:29"),
    np.datetime64("2024-09-07T16:16:03"),
    np.datetime64("2024-09-07T16:56:29"),
]

for t in specMACS_start_times:
    ax_wales.axvline(
        t,
        -0.1,
        0.0,
        color="k",
        alpha=0.25,
        linewidth=2.0,
        clip_on=False,
    )

dol_sondes_launch_times = dol_sondes_flight_day.launch_time.values
dol_sondes_launch_times = dol_sondes_launch_times[
    (dol_sondes_launch_times >= segment_start)
    & (dol_sondes_launch_times <= segment_end)
]

ax_wales.scatter(
    dol_sondes_launch_times,
    np.full(dol_sondes_launch_times.shape, 0),  # axes fraction
    transform=ax_wales.get_xaxis_transform(),
    color="k",
    marker="v",
    s=50,
    clip_on=False,
)

sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/figures_itcz_doldrums.png", bbox_inches="tight")

# %%
