# %%

from orcestra import get_flight_segments
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator

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
flight_id = "HALO-20240822a"
flight = flights[flight_id]

flight_start, flight_end = flight["takeoff"], flight["landing"]
flight_date = flight["date"]

segment_for_plot = [s for s in flight["segments"] if s["name"] == "atr_circle"][0]
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
wvel_circle_sel = circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).wvel

# %%

# WALES dataset
ds_wales_no_wv = open_dataset("wales").sel(time=slice(flight_start, flight_end))
ds_wales_no_wv = ds_wales_no_wv.where(ds_wales_no_wv.bsrgl_flags == 0)

# %%
# HAMP dataset
ds_hamp_flight = open_dataset("hamp_radar").sel(time=slice(flight_start, flight_end))

ds_hamp_passiv_flight = open_dataset("hamp_radiometer").sel(
    time=slice(flight_start, flight_end)
)
ds_hamp_passiv_segment = ds_hamp_passiv_flight.sel(
    time=slice(segment_start, segment_end)
)

# %%
# LAM dataset
time = f"{flight_date} 16:00"
lam_ds = open_dataset("lam")["prw"].sel(time=time)
ds_lam_sel = lam_ds.compute()

# %%
# Vertical velocity in the lower to mid-tropsphere in the circle segment

wvel_circle_sel_lmt = wvel_circle_sel.sel(altitude=slice(1.5, 8))
wvel_circle_sel_lmt_max_alt = wvel_circle_sel_lmt.isel(
    altitude=wvel_circle_sel_lmt.argmax()
).altitude.values
wvel_circle_sel_lmt_min_alt = wvel_circle_sel_lmt.isel(
    altitude=wvel_circle_sel_lmt.argmin()
).altitude.values
wvel_circle_sel_lmt_0 = wvel_circle_sel_lmt.isel(
    altitude=np.abs(wvel_circle_sel_lmt).argmin()
).altitude.values
print(
    f"Max vertical velocity in the lower to mid-troposphere: {wvel_circle_sel_lmt.max().values:.3f} m/s at {wvel_circle_sel_lmt_max_alt:.2f} km"
)
print(
    f"Min vertical velocity in the lower to mid-troposphere: {wvel_circle_sel_lmt.min().values:.3f} m/s at {wvel_circle_sel_lmt_min_alt:.2f} km"
)
print(f"Zero crossing altitude: {wvel_circle_sel_lmt_0:.3f} km")

# %%

iwv_crit = 48.0  # kg/m^2

ds_low_iwv = sondes_ds.where(sondes_ds.iwv < iwv_crit, drop=True)
low_iwv_sondes_id = ds_low_iwv.vaisala_serial_id

iwv_mask = sondes_ds.vaisala_serial_id.isin(low_iwv_sondes_id.values)
low_iwv_sondes = sondes_ds.where(iwv_mask, drop=True)

try:
    sondes_flight_day = sondes_ds.sel(launch_time=flight_date)
    print(
        f"Number of sondes launched on {flight_date}: {len(sondes_flight_day.launch_time)}"
    )
except KeyError:
    print("No sondes launched on this day.")

try:
    dol_sondes_flight_day = low_iwv_sondes.sel(launch_time=flight_date)
    print(
        f"Number of low IWV sondes launched on {flight_date}: {len(dol_sondes_flight_day.launch_time)}"
    )
except KeyError:
    print("No low IWV sondes on this day.")


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

vmin, vmax, vcenter = 30, 70, 48
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = plt.cm.Blues.copy()
im_les = egh.healpix_show(ds_lam_sel, cmap=cmap, norm=norm, ax=ax_lam, alpha=1.0)

egh.healpix_contour(
    ds_lam_sel,
    ax=ax_lam,
    levels=[48],
    colors="k",
    linewidths=0.5,
    zorder=10,
)

ax_lam.apply_aspect()
pos = ax_lam.get_position()

pos_ds = ax_ds.get_position()
ax_ds.set_position([pos_ds.x0, pos.y0, pos_ds.width, pos.height])

cbar_width = 0.01
cax_lam = fig.add_axes([pos.x1 + 0.015, pos.y0, cbar_width, pos.height])

cb = fig.colorbar(
    im_les,
    cax=cax_lam,
    orientation="vertical",
    label=rf"{ds_lam_sel.long_name} / {ds_lam_sel.units}",
)

mask = (tracks.time >= segment_start) & (tracks.time <= segment_end)

ax_lam.plot(
    ds_hamp_passiv_flight.lon,
    ds_hamp_passiv_flight.lat,
    c="k",
    linewidth=8,
    zorder=30,
)

ax_lam.scatter(
    ds_hamp_passiv_flight.lon,
    ds_hamp_passiv_flight.lat,
    c=ds_hamp_passiv_flight.IWV,
    cmap=cmap,
    norm=norm,
    s=10,
    zorder=31,
)

ax_lam.plot(
    tracks.lon.where(mask),
    tracks.lat.where(mask),
    color="C1",
    zorder=32,
)


ax_lam.set_xlabel("longitude / °")
ax_lam.set_ylabel("latitude / °")
ax_lam.spines[["right", "top"]].set_visible(False)
ax_lam.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)
ax_lam.set_title(flight_id)

# cmap_wales = plt.get_cmap("Blues").copy()
# cmap_wales.set_under((1, 1, 1, 0))  # transparent

from matplotlib.colors import LinearSegmentedColormap

cmap_wales = LinearSegmentedColormap.from_list(
    "WALES_BSR",
    """#cce5ff
    #99ccff #7399ff #4d66ff #264cd9 #0033b3 #005993 #008073 #008c39 #009900 #20a620 #40b340
    #81cd43 #c2e847 #e0f323 #ffff00 #ffef00 #ffe000 #ffc426 #ffa84d #ff8833 #ff691a #f2470d
    #e62600 #d21300 #bf0000 #ac0000 #990000 #4c0000""".split(),
)
cmap_wales.set_under((1, 1, 1, 0))  # transparent

cmap_wales_ticklocator = FixedLocator(
    [1, 1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0, 30.0, 50.0, 70.0, 90.0]
)

ds_wales_sel_segment = ds_wales_no_wv.sel(time=slice(segment_start, segment_end))
ds_wales_sel_segment = ds_wales_sel_segment.assign_coords(
    altitude=ds_wales_sel_segment.altitude / 1e3
)

im_wales = ds_wales_sel_segment["bsrgl"].plot(
    alpha=0.9,
    x="time",
    #    levels=np.arange(0, 81, 3),
    ax=ax_wales,
    cmap=cmap_wales,
    norm=colors.LogNorm(vmin=1, vmax=100),
    add_colorbar=False,
)


# HAMP radar reflectivity

cmap = plt.get_cmap("pink_r").copy()
cmap.set_under((1, 1, 1, 0))  # transparent

reflectivity_segment = ds_hamp_flight.radar_reflectivity.sel(
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


cb = fig.colorbar(
    im_hamp,
    cax=cax_wv,
    orientation="vertical",
    label=r"$Z_e$ / mm$^6$ m$^{-3}$",
)

ax_wales.axhline(
    wvel_circle_sel_lmt_max_alt, color="k", linestyle="--", linewidth=1, zorder=0
)


cax_wales = fig.add_axes([pos.x1 + 0.095, pos.y0, cbar_width, pos.height])
cb_wales = fig.colorbar(
    im_wales,
    cax=cax_wales,
    orientation="vertical",
    label=r"total backscatter ratio / ",
)


wvel_circle_sel.plot(y="altitude", ax=ax_ds, label="Vertical velocity", color="C1")

cwv_circle_in_segment = circles_flight_day.sel(
    circle_id=segment_for_plot["segment_id"]
).iwv_mean.values

ax_ds.axvline(0, color="k", linestyle=":", linewidth=1, zorder=0)

ax_ds.set_xlabel("vertical velocity / m s$^{-1}$")
ax_ds.set_ylabel("height / km")
ax_ds.set_title(" ")
ax_ds.set_ylim(ymin=0, ymax=13)
ax_ds.set_xlim(xmin=-0.075, xmax=0.075)

ax_wales.set_ylim(ymin=0.5, ymax=11)

# plt.suptitle(flight_id)

sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/figures_itcz_outside.png", bbox_inches="tight")

# %%

print(
    f"Mean IWV in circle segment: {ds_hamp_passiv_segment.IWV.mean().values:.2f} kg/m^2"
)
print(
    f"Min IWV in circle segment: {ds_hamp_passiv_segment.IWV.min().values:.2f} kg/m^2"
)
print(
    f"Max IWV in circle segment: {ds_hamp_passiv_segment.IWV.max().values:.2f} kg/m^2"
)

(ds_hamp_passiv_segment.IWV.dropna(dim="time") < 50.00).sum() / len(
    ds_hamp_passiv_segment.IWV.dropna(dim="time")
)

# %%

plt.scatter(
    ds_hamp_passiv_segment.lat,
    ds_hamp_passiv_segment.IWV,
    c=ds_hamp_passiv_segment.time.values,
    cmap="viridis",
)
plt.colorbar()
# %%

plt.scatter(
    ds_hamp_passiv_segment.lat,
    ds_hamp_passiv_segment.IWV,
    c=ds_hamp_passiv_segment.lon.values,
    cmap="viridis",
)
plt.colorbar()
# %%

ds_hamp_passiv_segment.IWV.plot()
# %%

from cartopy.geodesic import Geodesic

geod = Geodesic()


def window_distance(ds, t0, t1):
    p0 = ds.sel(time=t0, method="nearest")
    p1 = ds.sel(time=t1, method="nearest")
    # inverse() returns [[distance_m, azimuth_start, azimuth_end]]
    dist_m = geod.inverse(
        (float(p0.lon), float(p0.lat)), (float(p1.lon), float(p1.lat))
    )[0, 0]
    return dist_m


# %%

t0 = ds_hamp_passiv_segment.time.isel(time=0).values
t1 = ds_hamp_passiv_segment.time.isel(time=0).values + np.timedelta64(3, "m")

delta_t_min = 3
tslices = [
    (
        ds_hamp_passiv_segment.time.isel(time=0).values,
        ds_hamp_passiv_segment.time.isel(time=0).values
        + np.timedelta64(delta_t_min, "m"),
    ),
    (
        ds_hamp_passiv_segment.time.isel(time=-1).values
        - np.timedelta64(delta_t_min, "m"),
        ds_hamp_passiv_segment.time.isel(time=-1).values,
    ),
]

for t0, t1 in tslices:
    print(f"Time window: {t0} to {t1}")
    ds_hamp_passiv_segment.IWV.sel(time=slice(t0, t1)).plot()
    window_dist_km = window_distance(ds_hamp_passiv_segment, t0, t1) / 1000
    window_dist_deg = window_dist_km / 110
    window_cwv_delta = (
        ds_hamp_passiv_segment.IWV.max() - ds_hamp_passiv_segment.IWV.min()
    )

    print(f"Distance covered in first 3 minutes: {window_dist_km:.2f} km")
    print(f"Change in CWV in first 3 minutes: {window_cwv_delta:.2f} mm")
    print(
        f"Gradient of IWV in first 3 minutes: {(window_cwv_delta / window_dist_deg):.2f} mm/°"
    )

# %%
