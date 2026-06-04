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
flight_id = "HALO-20240907a"
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

tracks = all_tracks.sel(time=flight_date)

# %%
# Dropsonde data
sondes_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)

circles_flight_day = sondes_ds.sel(circle_time=flight_date)

sondes_ds = sondes_ds.swap_dims({"circle": "circle_id"})

wsp_crit = 3.0  # m/s
wsp_sfc = sondes_ds.wspd.sel(altitude=slice(0, 100)).mean("altitude")

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
ds_wales_wv = xr.open_dataset(store, engine="zarr")
ds_wales_wv = ds_wales_wv.sel(time=slice(flight_start, flight_end))

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
lam_ds = lam_ds.assign(sfcwind=lambda dx: np.hypot(dx.uas, dx.vas))
time = f"{flight_date} 16:00"

ds_lam_sel = lam_ds["sfcwind"].sel(time=time).compute()

# %%

scatter_kwargs = {
    "s": 25,
    "edgecolor": "k",
    "linewidths": 1.0,
}


fig = plt.figure(figsize=(12, 8))

gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[4, 1])

ax_lam = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_ds = fig.add_subplot(gs[0, 1])

ax_wales = fig.add_subplot(gs[1, :])

base_map(coastline_kwargs={"color": "k"}, ax=ax_lam)

vmin, vmax, vcenter = 0, 12, 3
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = plt.cm.Spectral_r
im = egh.healpix_show(ds_lam_sel, cmap=cmap, norm=norm, ax=ax_lam, alpha=1.0)

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
    label="Launch",
    alpha=0.5,
    **scatter_kwargs,
)

ax_lam.scatter(
    dol_sondes_flight_day.launch_lon,
    dol_sondes_flight_day.launch_lat,
    color=None,
    label="Launch",
    alpha=1.0,
    zorder=10,
    facecolors="white",
    **scatter_kwargs,
)

ax_lam.set_xlabel("longitude / °")
ax_lam.set_ylabel("latitude / °")
ax_lam.spines[["right", "top"]].set_visible(False)
ax_lam.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)
ax_lam.set_title(flight_id)


ds_wales_sel_segment = ds_wales_wv.sel(time=slice(segment_start, segment_end))
im_wales = ds_wales_sel_segment["RH"].plot(
    alpha=0.9,
    x="time",
    vmin=0,
    vmax=100,
    ax=ax_wales,
    cmap="YlGnBu",
    add_colorbar=False,
)

pos = ax_wales.get_position()

cax_wv = fig.add_axes(
    [
        pos.x0 + 0.25 * pos.width,  # left
        pos.y1 - 0.38,  # bottom
        0.5 * pos.width,  # width
        0.015,  # height
    ]
)

cb = fig.colorbar(
    im_wales,
    cax=cax_wv,
    orientation="horizontal",
    label=r"RH / %",
    shrink=0.75,
)


circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).wvel.plot(
    y="altitude", ax=ax_ds, label="Vertical velocity", color="k"
)

ax_ds.set_xlabel("vertical velocity / m s$^{-1}$")
ax_ds.set_ylabel("height / m")
ax_ds.set_title(" ")
ax_ds.set_yticklabels([])
ax_ds.set_ylim(ymin=0, ymax=13e3)
ax_ds.set_xlim(xmin=-0.075, xmax=0.075)


ax_wales.set_ylim(ymin=0, ymax=13e3)

sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/figures_itcz_doldrums.png", bbox_inches="tight")

# %%
