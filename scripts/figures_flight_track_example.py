# %%

"""Generate image with flight track superimposed on GOES visible image"""

from orcestra import get_flight_segments
from orcestra.flightplan import LatLon
from orcestra.weathermaps import goes_overlay
import numpy as np
from scipy.stats import binned_statistic_2d
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from percusion.utils import (
    base_map,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    halo_plot_dic,
    event_kinds_dic,
    get_halo_position,
)

import percusion
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]


sns.set_context("paper")

# %%
all_tracks = get_halo_position()

# %%

flights = get_flight_segments()["HALO"]

flight_name = "HALO-20240903a"  # "HALO-20240907a"  # "HALO-20240811a"
flight = flights[flight_name]
flight_date = flight["date"]

tracks = all_tracks.sel(time=flight_date)
plan = LatLon(lat=tracks["lat"], lon=tracks["lon"], label=flight_name)

sat_date_time = flight_date + "T15:20"
loc_at_sat = tracks.assign_coords({"tid": tracks.time}).sel(
    time=sat_date_time, method="nearest"
)

flight_date_no_dash = flight_date.replace("-", "")

# %%

events = {}
events["meteor_overpass"] = [
    e for e in flight["events"] if e["kinds"] == ["meteor_overpass"]
]
events["ec_underpass"] = [e for e in flight["events"] if e["kinds"] == ["ec_underpass"]]

# %%
# Advanced Microwave Scanning Radiometer (AMSR)
# Integrated water vapor in atmospheric column (TCWV) in mm
# https://nsidc.org/data/au_rain/versions/1

files = glob(f"{PROJECT_ROOT}/data/AMSR_U2_L2_Rain_V02_{flight_date_no_dash}*.he5")
if len(files) == 0:
    raise FileNotFoundError(
        f"No AMSR data files found. Please download data from "
        "https://nsidc.org/data/data-access-tool/AU_Rain/versions/1"
    )
if len(files) > 1:
    print(f"Warning: Found {len(files)} files, using first one")

file_path = files[0]

with h5py.File(file_path, "r") as f:
    swath = f["HDFEOS"]["SWATHS"]["AMSR2_L1R"]
    data = swath["Data Fields"]
    geo = swath["Geolocation Fields"]

    lat = geo["Latitude"][:]
    lon = geo["Longitude"][:]
    tcwv = data["TotalColWaterVapor"][:]

lon_flat = lon.flatten()
lat_flat = lat.flatten()
tcwv_flat = tcwv.flatten()

lon_grid = np.linspace(lon_min, lon_max, 250)
lat_grid = np.linspace(lat_min, lat_max, 250)

tcwv_grid, lon_edges, lat_edges, _ = binned_statistic_2d(
    lon_flat, lat_flat, tcwv_flat, statistic="mean", bins=[lon_grid, lat_grid]
)

# %%

fig, ax = base_map(coastline_kwargs={"color": "k"})
goes_overlay(sat_date_time, ax)
cax = ax.inset_axes([1.02, 0.0, 0.015, 1])

overlays = []

# %%

for a in reversed(overlays):
    a.remove()
overlays.clear()
cax.clear()

halo_plot_dic["alpha"] = 1.0

(track,) = ax.plot(plan.lon, plan.lat, **halo_plot_dic)
overlays.append(track)

from collections import defaultdict

scatter_handles = defaultdict(list)

for key, event in events.items():
    for e in event:
        t = e["time"]
        lon, lat = plan.lon.sel(time=t, method="nearest"), plan.lat.sel(
            time=t, method="nearest"
        )

        h = ax.scatter(
            lon,
            lat,
            **event_kinds_dic[key],
            zorder=3,
            linewidths=3,
        )
        scatter_handles[key].append(h)
        overlays.append(h)


# Satellite swath

cf = ax.contourf(
    lon_grid[:-1] + np.diff(lon_grid) / 2,
    lat_grid[:-1] + np.diff(lat_grid) / 2,
    tcwv_grid.T,
    extent=[lon_min, lon_max, lat_min, lat_max],
    cmap="Blues",
    alpha=0.75,
    levels=np.arange(36, 68, 4),
)
overlays.append(cf)

fig.colorbar(
    cf,
    label="CWV / mm",
    cax=cax,
    orientation="vertical",
)

fig.canvas.draw_idle()
fig

# %%

fig.savefig(
    f"{PROJECT_ROOT}/figures/{flight_name}_flight_track.pdf", bbox_inches="tight"
)
# %%
