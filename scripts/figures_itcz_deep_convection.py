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
# LAM dataset
url = "https://eerie.cloud.dkrz.de/datasets/orcestra_1250m_2d_hpz12/kerchunk"
lam_ds = xr.open_dataset(url, chunks={}, engine="zarr", zarr_format=3)

# %%
# Dropsonde data
sondes_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)
sondes_ds = sondes_ds.swap_dims({"circle": "circle_id"})
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

gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[4, 1])

ax_sat = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_ds = fig.add_subplot(gs[0, 1])

ax_hamp = fig.add_subplot(gs[1, :])

base_map(coastline_kwargs={"color": "w"}, ax=ax_sat)
im = egh.healpix_show(ds_lam_sel, ax=ax_sat, alpha=1.0, cmap="Greys_r")

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
    )  # , label=s["name"])

for k in ["circle", "straight_leg", "ec_track", "atr_coordination"]:
    ax_sat.plot([], [], color=kinds2color(k), label=k)


ax_sat.set_xlabel("longitude / °")
ax_sat.set_ylabel("latitude / °")
ax_sat.spines[["right", "top"]].set_visible(False)
ax_sat.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)
ax_sat.set_title(flight_id)

ds_hamp_sel_segment = ds_hamp_flight.sel(
    time=slice(time_plot_hamp_start, time_plot_hamp_end)
)
im_hamp = ds_hamp_sel_segment["radar_reflectivity"].plot(
    norm=colors.LogNorm(),
    alpha=0.9,
    x="time",
    vmin=1e-5,
    vmax=1e5,
    ax=ax_hamp,
    cmap="YlGnBu",
    add_colorbar=False,
)

pos = ax_hamp.get_position()

cax_wv = fig.add_axes(
    [
        pos.x0 + 0.25 * pos.width,  # left
        pos.y1 - 0.38,  # bottom
        0.5 * pos.width,  # width
        0.015,  # height
    ]
)

cb = fig.colorbar(
    im_hamp,
    cax=cax_wv,
    orientation="horizontal",
    label=r"equivalent reflectivity factor / mm$^6$ m$^{-3}$",
    shrink=0.75,
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

circles_flight_day.sel(circle_id="HALO-20240903a_c30f").wvel.plot(
    y="altitude", ax=ax_ds, label="Vertical velocity", color=kinds2color("circle")
)
ax_ds.set_xlabel("vertical velocity / m s$^{-1}$")
ax_ds.set_ylabel("height / m")
ax_ds.set_title(" ")
ax_ds.set_yticklabels([])
ax_ds.set_ylim(ymin=0, ymax=13e3)
ax_ds.set_xlim(xmin=-0.075, xmax=0.075)

sns.despine()

plt.savefig(
    f"{PROJECT_ROOT}/figures/figures_itcz_deep_convection.png", bbox_inches="tight"
)

# %%
