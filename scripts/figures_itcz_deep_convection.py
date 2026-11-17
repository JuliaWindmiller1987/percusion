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
from percusion.utils import base_map, kinds2color

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%

flights = get_flight_segments()["HALO"]
flight_id = "HALO-20240903a"
flight = flights[flight_id]

flight_start, flight_end = flight["takeoff"], flight["landing"]
flight_date = flight["date"]

segment_for_plot = [s for s in flight["segments"] if s["name"] == "circle_mid"][0]
segment_start = np.datetime64(segment_for_plot["start"])
segment_end = np.datetime64(segment_for_plot["end"])

# %%
# BAHAMAS dataset
all_tracks = xr.open_dataset(
    "ipfs://bafybeias3h5uxtt4ky4d4gn6l6gxjqfkzbde5jlunya6g3umnkvn7xoyoe", engine="zarr"
)
flight_date = flight_id[5:9] + "-" + flight_id[9:11] + "-" + flight_id[11:13]

tracks = all_tracks.sel(time=flight_date)

# %%
# HAMP dataset
ds_hamp = xr.open_dataset(
    "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a", engine="zarr"
)
ds_hamp_flight = ds_hamp.sel(time=slice(flight_start, flight_end))

# %%
# WALES dataset
store = (
    "https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/wales/wales_wv.zarr"
)
ds_wales_wv = xr.open_dataset(store, engine="zarr")
ds_wales_wv = ds_wales_wv.sel(time=slice(flight_start, flight_end))
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
# LAM dataset
url = "https://eerie.cloud.dkrz.de/datasets/orcestra_1250m_2d_hpz12/kerchunk"
lam_ds = xr.open_dataset(url, chunks={}, engine="zarr", zarr_format=3)

# %%
# Dropsonde data
sondes_ds = xr.open_dataset(
    "ipfs://bafybeibgeeqs5uhmbqy4hz4v3pihrfisiklcetisxkf63d2r473szaprwi", engine="zarr"
)
sondes_ds = sondes_ds.swap_dims({"circle": "circle_id"})
sondes_ds = sondes_ds.assign_coords(altitude=sondes_ds.altitude / 1e3)
sondes_flight_day = sondes_ds.sel(launch_time=flight_date)
circles_flight_day = sondes_ds.sel(circle_time=flight_date)

center_circle_id = "HALO-20240903a_c30f"

# %%

ec_underpass_events = [e for e in flight["events"] if "ec_underpass" in e["kinds"]]
meteor_overpass_events = [
    e for e in flight["events"] if "meteor_overpass" in e["kinds"]
]

lam_time = f"{flight_date} 15:20"  # ec_underpass_events[0]["time"]
ds_lam_sel = lam_ds["rsut"].sel(time=lam_time, method="nearest").compute()

print(ds_lam_sel.time.values, lam_time)

# %%

time_delta = np.timedelta64(20, "m")
ec_start, ec_end = ec_underpass_events[0]["time"] - np.timedelta64(
    20, "m"
), ec_underpass_events[0]["time"] + np.timedelta64(20, "m")


fig = plt.figure(figsize=(12, 8))

gs = fig.add_gridspec(
    2, 2, height_ratios=[1.5, 1], width_ratios=[4, 1], hspace=0.37, wspace=0.15
)

ax_sat = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_ds = fig.add_subplot(gs[0, 1])

ax_hamp = fig.add_subplot(gs[1, :])

base_map(coastline_kwargs={"color": "w"}, ax=ax_sat)
im = egh.healpix_show(ds_lam_sel, ax=ax_sat, alpha=1.0, cmap="Greys_r")

pos = ax_sat.get_position()
width_scale = 0.8

cax_lam = fig.add_axes(
    [
        pos.x0 + pos.width * (1 - width_scale) / 2,  # left
        pos.y1 - 0.45,  # bottom
        pos.width * width_scale,  # width
        0.015,  # height
    ]
)

cb_lam = fig.colorbar(
    im,
    cax=cax_lam,
    orientation="horizontal",
    label=rf"outgoing shortwave radiation / {ds_lam_sel.units}",
)

segment_for_hamp = "circle_mid"

for s in flight["segments"]:
    t = slice(s["start"], s["end"])

    linestyle = "-"

    if s["name"] in ["circle_south", "circle_north"]:
        linestyle = "--"

    if s["name"] == segment_for_hamp:
        time_plot_hamp_start = s["start"]
        time_plot_hamp_end = s["end"]

    ax_sat.plot(
        tracks.lon.sel(time=t),
        tracks.lat.sel(time=t),
        c=kinds2color(s["kinds"]),
        linestyle=linestyle,
        linewidth=3.0,
    )  # , label=s["name"])

for k in ["circle", "straight_leg", "ec_track", "atr_coordination"]:
    ax_sat.plot([], [], color=kinds2color(k), label=k)


ax_sat.set_xlabel("longitude / °")
ax_sat.set_ylabel("latitude / °")
ax_sat.spines[["right", "top"]].set_visible(False)
ax_sat.set_title(flight_id)
ax_sat.legend(
    loc="upper left",
    ncol=1,
    frameon=True,
    edgecolor="k",
    labelcolor="white",
    fontsize=9,
    facecolor="k",
    framealpha=0.5,
)

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
    ax=ax_hamp,
    cmap=cmap_wales,
    add_colorbar=False,
)

ds_hamp_sel_segment = ds_hamp_flight.sel(
    time=slice(time_plot_hamp_start, time_plot_hamp_end)
)

cmap = plt.get_cmap("pink_r").copy()
cmap.set_under((1, 1, 1, 0))  # transparent

reflectivity_segment = ds_hamp_sel_segment["radar_reflectivity"]
reflectivity_segment = reflectivity_segment.assign_coords(
    altitude=reflectivity_segment.altitude / 1e3
)

im_hamp = reflectivity_segment.plot(
    y="altitude",
    ax=ax_hamp,
    label=f"IWV bin {bin}",
    cmap=cmap,
    norm=colors.LogNorm(vmin=1e-5, vmax=1e5),
    add_colorbar=False,
)

pos = ax_hamp.get_position()
ax_hamp.set_position([pos.x0, pos.y0, pos.width * 0.82, pos.height])
pos = ax_hamp.get_position()

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

ax_hamp.axvline(
    ec_underpass_events[0]["time"],
    0,
    0.1,
    color=kinds2color("ec_track"),
    linestyle=":",
    label="EC Underpass",
)

for event in meteor_overpass_events:
    ax_hamp.axvline(
        event["time"], 0, 0.1, color=kinds2color("ec_track"), label="METEOR Overpass"
    )
    print(f"METEOR Overpass at {event['time']}")

ax_hamp.set_ylim(ymin=0)

circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).wvel.plot(
    y="altitude", ax=ax_ds, label="Vertical velocity", color=kinds2color("circle")
)

ax_ds.axvline(0, color="k", linestyle=":", linewidth=1, zorder=0)

ax_ds.set_xlabel("vertical velocity / m s$^{-1}$")
ax_ds.set_ylabel("height / km")
ax_ds.set_title(" ")
ax_ds.set_ylim(ymin=0, ymax=13)
ax_ds.set_xlim(xmin=-0.075, xmax=0.075)

cwv_circle_in_segment = circles_flight_day.sel(
    circle_id=segment_for_plot["segment_id"]
).iwv_mean.values

cwv_circle_in_segment = circles_flight_day.sel(
    circle_id=segment_for_plot["segment_id"]
).iwv_mean.values

velox_start_time = np.datetime64("2024-09-03T15:10:00")
velox_end_time = np.datetime64("2024-09-03T15:20:00")

ax_hamp.axvspan(
    velox_start_time,
    velox_end_time,
    -0.025,
    0.025,
    color="k",
    alpha=0.25,
    clip_on=False,
)


print(f"Mean IWV in circle segment: {cwv_circle_in_segment:.2f} kg/m^2")

sns.despine()

plt.savefig(
    f"{PROJECT_ROOT}/figures/figures_itcz_deep_convection.png", bbox_inches="tight"
)

# %%
