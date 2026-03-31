# %%
from orcestra import get_flight_segments
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from orcestra.flightplan import sal, tbpb
import numpy as np

import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import base_map, list_of_kinds, get_halo_position
import pandas as pd

# %%

import percusion
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%


def close_path(lon, lat):
    """Close the path by appending the first point to the end."""
    lon_closed = np.append(lon, lon[0])
    lat_closed = np.append(lat, lat[0])
    return lon_closed, lat_closed


# %%

flights = get_flight_segments()["HALO"]
flights.pop("HALO-20240809b", None)
flights.pop("HALO-20240929a", None)
flight_ids = list(flights.keys())

print(f"Total number of flights: {len(flight_ids)}")
# %%

ds = get_halo_position()

campaign_start = flights[flight_ids[0]]["takeoff"]
campaign_end = flights[flight_ids[-1]]["landing"]
ds = ds.sel(time=slice(campaign_start, campaign_end))

# %%

tcwv_era5 = xr.open_dataset(f"{PROJECT_ROOT}/data/cwvEra520240809.nc")
land_sea_mask_era5 = xr.open_dataset(f"{PROJECT_ROOT}/data/landSeaMaskEra520240809.nc")

tcwv_era5 = tcwv_era5.where(land_sea_mask_era5.lsm == 0, drop=True)

# %%

segments = [
    {**s, "flight_id": flight_id}
    for flight_id, flight in flights.items()
    for s in flight["segments"]
]

list_of_kinds(segments)
# %%

events = [
    {**e, "flight_id": flight_id}
    for flight_id, flight in flights.items()
    for e in flight["events"]
]

list_of_kinds(events)

# %%

fig, ax = base_map(coastline_kwargs={"color": "k"})

cmap = plt.get_cmap("Blues")
norm = mcolors.Normalize(vmin=36, vmax=68)

tcwv_levels = [48]
tcwv_colors = [cmap(norm(level)) for level in tcwv_levels]

tcwv_era5.mean("valid_time").tcwv.plot.contour(
    ax=ax, linewidths=4, levels=tcwv_levels, colors=tcwv_colors, zorder=1
)

halo_plot_dic = {
    "color": "#F2935C",
    "alpha": 0.75,
    "linewidth": 1.5,
    "label": "HALO track",
    "zorder": 2,
}
plt.plot(ds.lon, ds.lat, **halo_plot_dic)

segment_kind_dic = {
    "circle": {
        "color": "#F2935C47",
        "alpha": 0.2,
        "label": "Circle",
        "compound": {
            "kind": "atr_coordination",
            "color": "#B38970",
            "alpha": 0.5,
            "label": "ATR circle",
        },
    },
}
event_kinds_dic = {
    "ec_underpass": {
        "marker": "x",
        "s": 100,
        "color": "#709D9D",
        "label": "EC underpass",
    },
    "meteor_overpass": {
        "marker": "x",
        "s": 100,
        "color": "#736A65",
        "label": "METEOR overpass",
    },
}


for f in flight_ids:
    for s in segments:
        for segment_kind in segment_kind_dic.keys():

            segment_props = segment_kind_dic[segment_kind]
            compound_props = segment_props.get("compound", None)

            if segment_kind in s["kinds"] and f == s["flight_id"]:
                t = slice(s["start"], s["end"])

                lon, lat = ds.lon.sel(time=t), ds.lat.sel(time=t)
                lon_closed, lat_closed = close_path(lon, lat)

                if compound_props["kind"] in s["kinds"]:
                    ax.fill(
                        lon_closed,
                        lat_closed,
                        color=compound_props["color"],
                        edgecolor=None,
                        alpha=compound_props["alpha"],
                        label=compound_props["label"],
                        zorder=9,
                    )

                else:
                    ax.fill(
                        lon_closed,
                        lat_closed,
                        color=segment_props["color"],
                        edgecolor=None,
                        alpha=segment_props["alpha"],
                        label=segment_props["label"],
                        zorder=10,
                    )

    for e in events:
        for event_kind in event_kinds_dic.keys():

            event_props = event_kinds_dic[event_kind]

            if event_kind in e["kinds"] and f == e["flight_id"]:
                t = e["time"]
                lon, lat = ds.lon.sel(time=t, method="nearest"), ds.lat.sel(
                    time=t, method="nearest"
                )

                ax.scatter(
                    lon,
                    lat,
                    color=event_props["color"],
                    label=event_props["label"],
                    marker=event_props["marker"],
                    s=event_props["s"],
                    zorder=11,
                )


# Create legend handles
handles = []
labels = []

# Add HALO track line to legend
handles.append(plt.Line2D([0], [0], **halo_plot_dic))
labels.append(halo_plot_dic["label"])

for props in event_kinds_dic.values():
    if props["label"] not in labels:
        handles.append(
            plt.scatter(
                [],
                [],
                marker=props["marker"],
                color=props["color"],
                s=props["s"],
            )
        )
        labels.append(props["label"])

for props in segment_kind_dic.values():
    if props["label"] not in labels:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=props["color"], alpha=props["alpha"])
        )
        labels.append(props["label"])

    compound_props = props.get("compound", None)
    if compound_props and compound_props["label"] not in labels:
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=compound_props["color"],
                alpha=compound_props["alpha"],
            )
        )
        labels.append(compound_props["label"])

for level, color in zip(tcwv_levels, tcwv_colors):
    handles.append(
        plt.Line2D([0], [0], color=color, linewidth=4, label=f"CWV {level} mm")
    )
    labels.append(f"CWV {level} mm")


ax.legend(handles, labels, loc="lower left", frameon=False)

ax.set_xlabel("longitude / °")
ax.set_ylabel("latitude / °")
ax.set_title(" ")

# ax.scatter(sal.lon, sal.lat, marker="o", color="k")
# ax.annotate("SAL", (sal.lon + 0.5, sal.lat + 0.5))
# ax.scatter(tbpb.lon, tbpb.lat, marker="o", color="k")
# ax.annotate("BARBADOS", (tbpb.lon + 0.5, tbpb.lat + 0.5))
ax.spines[["right", "top"]].set_visible(False)

plt.savefig(f"{PROJECT_ROOT}/figures/flight_overview.pdf", bbox_inches="tight")
# %%
