# %%

import cartopy.crs as ccrs
import easygems.healpix as egh
import intake
import numpy as np
import xarray as xr

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import base_map

import percusion
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

#
flight_name = "HALO-20240907a"  # "HALO-20240907a"  # "HALO-20240811a"

# %%
# HALO data
all_tracks = xr.open_dataset(
    "ipfs://bafybeias3h5uxtt4ky4d4gn6l6gxjqfkzbde5jlunya6g3umnkvn7xoyoe", engine="zarr"
)

flight_date = flight_name[5:9] + "-" + flight_name[9:11] + "-" + flight_name[11:13]

tracks = all_tracks.sel(time=flight_date)

# %%
# Dropsonde data
sondes_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)
sondes_ds = sondes_ds.swap_dims({"circle": "circle_id"})

wsp_crit = 3.0  # m/s
wsp_sfc = sondes_ds.wspd.sel(altitude=slice(0, 100)).mean("altitude")

ds_low_wsp = sondes_ds.where(wsp_sfc < wsp_crit, drop=True)
doldrum_sondes_id = ds_low_wsp.vaisala_serial_id

dol_mask = sondes_ds.vaisala_serial_id.isin(doldrum_sondes_id.values)
dol_sondes = sondes_ds.where(dol_mask, drop=True)

# %%

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
# LAM dataset
url = "https://eerie.cloud.dkrz.de/datasets/orcestra_1250m_2d_hpz12/kerchunk"
lam_ds = xr.open_dataset(url, chunks={}, engine="zarr", zarr_format=3)
lam_ds = lam_ds.assign(sfcwind=lambda dx: np.hypot(dx.uas, dx.vas))
time = f"{flight_date} 16:00"

ds_sel = lam_ds["sfcwind"].sel(time=time).compute()

# %%

scatter_kwargs = {
    "s": 50,
    "edgecolor": "k",
    "linewidths": 1.25,
}

fig, ax = base_map(coastline_kwargs={"color": "k"})

plt.plot(
    tracks.lon, tracks.lat, color="k", alpha=1.0, linewidth=1.5, label="HALO track"
)

plt.scatter(
    sondes_flight_day.launch_lon,
    sondes_flight_day.launch_lat,
    facecolors="none",
    label="Launch",
    alpha=1.0,
    **scatter_kwargs,
)

plt.scatter(
    dol_sondes_flight_day.launch_lon,
    dol_sondes_flight_day.launch_lat,
    color="k",
    label="Launch",
    alpha=1.0,
    zorder=10,
    facecolors="white",
    **scatter_kwargs,
)

vmin, vmax, vcenter = 0, 12, 3
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = plt.cm.Spectral_r
im = egh.healpix_show(ds_sel, cmap=cmap, norm=norm, ax=ax, alpha=1.0)

# Add colorbar
cax = make_axes_locatable(ax).append_axes(
    "right", size="1%", pad=0.1, axes_class=plt.Axes
)
cbar = fig.colorbar(
    im,
    cax=cax,
    ticks=range(0, 13, 3),
)

cbar.set_label(label="Horizontal surface wind speed [m/s]", size=12)

# Add title with formatted timestamp
ax.set_title(" ")

plt.savefig(
    f"{PROJECT_ROOT}/figures/doldrums.pdf",
    bbox_inches="tight",
)

# %%

mean_doldrum_count = xr.open_dataset(
    f"{PROJECT_ROOT}/data/mean_doldrum_freq_ocean_Lorcestra.nc"
)

# %%

fig, ax = base_map(coastline_kwargs={"color": "k"})
im = egh.healpix_show(
    mean_doldrum_count.low_wind_speed_frequency,
    cmap="GnBu_r",
    ax=ax,
    vmin=0,
    vmax=0.5,
    alpha=0.8,
)


fig.colorbar(
    im,
    label="Fraction of time with near-surface wind speed < 3 m/s",
    orientation="horizontal",
    pad=0.1,
    shrink=0.8,
)

plt.scatter(
    sondes_ds.launch_lon,
    sondes_ds.launch_lat,
    facecolors="none",
    label="Launch",
    alpha=1,
    zorder=9,
    **scatter_kwargs,
)

plt.scatter(
    dol_sondes.launch_lon,
    dol_sondes.launch_lat,
    facecolors="#F7B69157",
    alpha=1,
    label="Launch",
    zorder=10,
    **scatter_kwargs,
)
# %%
