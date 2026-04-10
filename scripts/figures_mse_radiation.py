# %%

from orcestra import get_flight_segments
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import percusion
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%
flights = get_flight_segments()["HALO"]
ds_hamp = xr.open_dataset(
    "ipfs://bafybeigmd3dovwm45ylfqxnn2jphsrdjl2jt3dfytv7grkyhleaq42jthe", engine="zarr"
)

ds_meteor = xr.open_dataset(
    "ipfs://bafybeib5awa3le6nxi4rgepn2mwxj733aazpkmgtcpa3uc2744gxv7op44", engine="zarr"
)

# %%

flight_id = "HALO-20240903a"
flight = flights[flight_id]

flight_start, flight_end = flight["takeoff"], flight["landing"]
# %%

ds_hamp_flight = ds_hamp.sel(time=slice(flight_start, flight_end))
ds_hamp_flight["Ze"].plot(norm=colors.LogNorm(), x="time")
# %%

ec_underpass_events = [e for e in flight["events"] if "ec_underpass" in e["kinds"]]
metero_overpass_events = [
    e for e in flight["events"] if "meteor_overpass" in e["kinds"]
]
# %%

time_delta = np.timedelta64(20, "m")
ec_start, ec_end = ec_underpass_events[0]["time"] - np.timedelta64(
    20, "m"
), ec_underpass_events[0]["time"] + np.timedelta64(20, "m")

ds_hamp_ec_underpass = ds_hamp_flight.sel(time=slice(ec_start, ec_end))

fig, ax = plt.subplots(figsize=(10, 3))
ds_hamp_ec_underpass["Zg"].plot(
    norm=colors.LogNorm(),
    alpha=0.9,
    x="time",
    vmin=1e-3,
    vmax=1e3,
    ax=ax,
    cmap="YlGnBu",
)
plt.axvline(ec_underpass_events[0]["time"], color="C0", label="EC Underpass")

for event in metero_overpass_events:
    plt.axvline(event["time"], color="C1", label="METEOR Overpass")
    print(f"METEOR Overpass at {event['time']}")

plt.ylim(ymin=0)
sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/mse_radiation.pdf", bbox_inches="tight")
