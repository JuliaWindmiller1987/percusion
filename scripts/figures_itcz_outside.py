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
from percusion.utils import base_map, kinds2color

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
all_tracks = xr.open_dataset(
    "ipfs://bafybeias3h5uxtt4ky4d4gn6l6gxjqfkzbde5jlunya6g3umnkvn7xoyoe", engine="zarr"
)

tracks = all_tracks.sel(time=flight_date)

# %%
# Dropsonde data
sondes_ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)

sondes_ds = sondes_ds.swap_dims({"circle": "circle_id"})
circles_flight_day = sondes_ds.sel(circle_time=flight_date)

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

# # "HALO-20240811a", "HALO-20240813a", "HALO-20240822a",

# for flight_id in ["HALO-20240903a", "HALO-20240907a"]:
#     flight = flights[flight_id]

#     flight_start, flight_end = flight["takeoff"], flight["landing"]
#     flight_date = flight["date"]

#     circles_flight_day = sondes_ds.sel(circle_time=flight_date)

#     segment_for_plot = [s for s in flight["segments"] if s["name"] == "circle_north"][0]

#     segment_start = np.datetime64(segment_for_plot["start"])
#     segment_end = np.datetime64(segment_for_plot["end"])

#     circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).wvel.plot(
#         y="altitude", label=flight_id
#     )

#     segment_sondes = sondes_ds.sel(launch_time=slice(segment_start, segment_end))

#     print(
#         f"Min IWV {segment_sondes.iwv.min().values}, max IWV {segment_sondes.iwv.max().values}"
#     )

#     print(
#         f"Mean IWV in circle segment: {circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).iwv_mean.values:.2f} kg/m^2"
#     )
# plt.axvline(0)

# ax_ds = plt.gca()
# ax_ds.set_xlim(xmin=-0.075, xmax=0.075)
# plt.legend(frameon=False)

# %%
# WALES dataset
store = "https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/wales/wales_no_wv.zarr"
ds_wales_no_wv = xr.open_dataset(store, engine="zarr")
ds_wales_no_wv = ds_wales_no_wv.sel(time=slice(flight_start, flight_end))
ds_wales_no_wv = ds_wales_no_wv.where(ds_wales_no_wv.bsrg_flags == 0)


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
lam_ds = lam_ds.assign(sfcwind=lambda dx: np.hypot(dx.uas, dx.vas))
time = f"{flight_date} 16:00"

# %%

ds_lam_sel = lam_ds["prw"].sel(time=time).compute()

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

vmin, vmax, vcenter = 30, 70, 48
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
    label="dropsonde",
    alpha=0.5,
    **scatter_kwargs,
)

ax_lam.scatter(
    dol_sondes_flight_day.launch_lon,
    dol_sondes_flight_day.launch_lat,
    color=None,
    label="doldrums dropsonde",
    alpha=0.75,
    zorder=10,
    facecolors="k",
    **scatter_kwargs,
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
im_wales = ds_wales_sel_segment["bsrg"].plot(
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


im_hamp = reflectivity_segment.plot(
    y="altitude",
    ax=ax_wales,
    label=f"IWV bin {bin}",
    cmap=cmap,
    norm=colors.LogNorm(vmin=1e-5, vmax=1e5),
    add_colorbar=False,
)


pos = ax_wales.get_position()

cax_wv_positions = [
    pos.x0 - 0.025,  # left
    pos.y1 - 0.38,  # bottom
    0.5 * pos.width,  # width
    0.015,  # height
]

cax_wv = fig.add_axes(cax_wv_positions)
cax_wales = fig.add_axes(cax_wv_positions + np.array([0.425, 0, 0, 0]))


cb = fig.colorbar(
    im_hamp,
    cax=cax_wv,
    orientation="horizontal",
    label=r"equivalent reflectivity factor / mm$^6$ m$^{-3}$",
    shrink=0.75,
)

cb_wales = fig.colorbar(
    im_wales,
    cax=cax_wales,
    orientation="horizontal",
    label=r"total backscatter ratio / ",
    shrink=0.75,
)

circles_flight_day.sel(circle_id=segment_for_plot["segment_id"]).wvel.plot(
    y="altitude", ax=ax_ds, label="Vertical velocity", color="k"
)

cwv_circle_in_segment = circles_flight_day.sel(
    circle_id=segment_for_plot["segment_id"]
).iwv_mean.values

# print(f"Mean IWV in circle segment: {cwv_circle_in_segment:.2f} kg/m^2")

ax_ds.set_xlabel("vertical velocity / m s$^{-1}$")
ax_ds.set_ylabel("height / m")
ax_ds.set_title(" ")
ax_ds.set_ylim(ymin=0, ymax=13e3)
ax_ds.set_xlim(xmin=-0.075, xmax=0.075)

ax_wales.set_ylim(ymin=500, ymax=13e3)

sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/figures_itcz_outside.png", bbox_inches="tight")

# %%


hamp_path_v2 = PROJECT_ROOT / "data" / "HAMP_with_IWV_IWP_LWP_TLWP_v2.nc"
ds_hamp = xr.open_dataset(hamp_path_v2)
hamp_orcestra = ds_hamp.sel(time=slice(segment_start, segment_end))

iwv_hamp_orcestra = hamp_orcestra["IWV"]

iwv_hamp_orcestra.plot()

# %%
