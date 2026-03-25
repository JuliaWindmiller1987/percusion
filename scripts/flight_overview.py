# %%
from orcestra import get_flight_segments
import xarray as xr
import matplotlib.pyplot as plt
from orcestra.flightplan import sal, tbpb
import numpy as np

import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import base_map, lon_min, lon_max, lat_min, lat_max

# %%


def close_path(lon, lat):
    """Close the path by appending the first point to the end."""
    lon_closed = np.append(lon, lon[0])
    lat_closed = np.append(lat, lat[0])
    return lon_closed, lat_closed


def list_of_kinds(list_of_dicts):

    lists_of_kinds = [
        dict["kinds"] if isinstance(dict["kinds"], (list)) else [dict["kinds"]]
        for dict in list_of_dicts
    ]
    kinds = [i for list_of_kinds in lists_of_kinds for i in list_of_kinds]

    return set(kinds)


def get_halo_position(freq="1s"):
    """Return the HALO position at a given time frequency."""
    # root = "ipns://latest.orcestra-campaign.org"
    root = "ipfs://QmWyyXuoTGJbf9MGSEjsfAdZzX8fWPfJByDgTb1yR9LWUg"
    return (
        xr.open_dataset(f"{root}/products/HALO/position_attitude.zarr", engine="zarr")
        .reset_coords(("lat", "lon"))[["lat", "lon"]]
        .resample(time=freq)
        .mean()
        .load()
    )


# %%

ds = get_halo_position()
# %%

meta = get_flight_segments()

# %%

flight_ids = [flight_id for flights in meta.values() for flight_id in flights]
print(f"Total number of flights: {len(flight_ids)}")

# %%

segments = [
    {**s, "platform_id": platform_id, "flight_id": flight_id}
    for platform_id, flights in meta.items()
    for flight_id, flight in flights.items()
    for s in flight["segments"]
]

list_of_kinds(segments)
# %%

events = [
    {**e, "platform_id": platform_id, "flight_id": flight_id}
    for platform_id, flights in meta.items()
    for flight_id, flight in flights.items()
    for e in flight["events"]
]

list_of_kinds(events)

# %%

fig, ax = base_map(coastline_kwargs={"color": "k"})

plt.plot(ds.lon, ds.lat, "k", alpha=0.1, linewidth=1, label="HALO")

kind = "circle"

for f in flight_ids:
    for s in segments:
        if kind in s["kinds"] and f == s["flight_id"]:
            t = slice(s["start"], s["end"])

            lon, lat = ds.lon.sel(time=t), ds.lat.sel(time=t)
            lon_closed, lat_closed = close_path(lon, lat)

            # ax.plot(lon, lat, c="C0", label=s["name"])
            ax.fill(lon_closed, lat_closed, color="C0", alpha=0.3)

ax.set_xlabel("longitude / °")
ax.set_ylabel("latitude / °")

ax.scatter(sal.lon, sal.lat, marker="o", color="k")
ax.annotate("SAL", (sal.lon, sal.lat))
ax.scatter(tbpb.lon, tbpb.lat, marker="o", color="k")
ax.annotate("BARBADOS", (tbpb.lon, tbpb.lat))
ax.spines[["right", "top"]].set_visible(False)


# %%
