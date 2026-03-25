# %%

"""Generate image with flight track superimposed on GOES visible image"""

from orcestra.flightplan import LatLon, path_preview, path_as_ds, plot_path
from orcestra.weathermaps import goes_overlay
import xarray as xr
import numpy as np
from scipy.stats import binned_statistic_2d
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cartopy.crs as ccrs

from percusion.utils import base_map, lon_min, lon_max, lat_min, lat_max

sns.set_context("paper")

# %%
all_tracks = xr.open_dataset(
    "ipfs://bafybeias3h5uxtt4ky4d4gn6l6gxjqfkzbde5jlunya6g3umnkvn7xoyoe", engine="zarr"
)

# %%

flight_name = "HALO-20240811a"  # "HALO-20240907a"  # "HALO-20240811a"
flight_date = flight_name[5:9] + "-" + flight_name[9:11] + "-" + flight_name[11:13]

tracks = all_tracks.sel(time=flight_date)
plan = LatLon(lat=tracks["lat"], lon=tracks["lon"], label=flight_name)

sat_date_time = flight_date + "T15:20"
loc_at_sat = tracks.assign_coords({"tid": tracks.time}).sel(
    time=sat_date_time, method="nearest"
)

flight_date_no_dash = flight_date.replace("-", "")
# %%
# Advanced Microwave Scanning Radiometer (AMSR)
# Integrated water vapor in atmospheric column (TCWV) in mm
# https://nsidc.org/data/au_rain/versions/1

files = glob(f"./data/AMSR_U2_L2_Rain_V02_{flight_date_no_dash}*.he5")
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

fig, ax = base_map()

plt.plot(plan.lon, plan.lat, "C1")
# plt.scatter(loc_at_sat.lon, loc_at_sat.lat, color="C1", )

goes_overlay(sat_date_time, ax)

# Satellite swath

plt.contourf(
    lon_grid[:-1] + np.diff(lon_grid) / 2,
    lat_grid[:-1] + np.diff(lat_grid) / 2,
    tcwv_grid.T,
    extent=[lon_min, lon_max, lat_min, lat_max],
    cmap="Blues",
    alpha=0.75,
    levels=np.arange(36, 68, 4),
)

plt.colorbar(
    label="CWV / mm",
    ax=ax,
    orientation="horizontal",
    pad=0.1,
    shrink=0.5,
)

plt.savefig(f"./figures/{flight_name}_flight_track.pdf", bbox_inches="tight")
# %%
