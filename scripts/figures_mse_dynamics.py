# %%
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

import importlib
import percusion.utils

importlib.reload(percusion.utils)
from percusion.utils import base_map, list_of_kinds, get_halo_position

# %%
import percusion
from pathlib import Path

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

# %%

ds = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)

# %%

# calculate moist static energy
# h = cp * T + g * z + L * q_v

T = ds.ta_mean  # temperature in K
z = ds.altitude  # altitude in m
q = ds.q_mean  # specific humidity in kg/kg
cp = 1005.7  # J/(kg*K)
g = 9.81  # m/s^2
L = 2.501e6  # J/kg

q_v = q / (1 - q)  # convert specific humidity to mixing ratio
h = cp * T + g * z + L * q_v
ds["h_mean"] = h / 1e3  # convert to kJ/kg
ds["h_mean"].attrs["long_name"] = "circle mean of moist static energy"
ds["h_mean"].attrs["units"] = "kJ/kg"

# %%

fig, ax = base_map(coastline_kwargs={"color": "k"})

plt.scatter(
    ds.launch_lon, ds.launch_lat, color="C0", label="Launch", s=20, edgecolor="k"
)
# %%
ymin, ymax = 0, 12.5e3

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for i, var in enumerate(["wvel", "h_mean"]):
    ds_var = ds[var].sel(altitude=slice(ymin, ymax))
    var_mean = ds_var.mean("circle")
    var_std = ds_var.std("circle")

    var_mean.plot(ax=ax[i], y="altitude", color="C1")
    ax[i].fill_betweenx(
        var_mean.altitude,
        var_mean - var_std,
        var_mean + var_std,
        color="C1",
        alpha=0.3,
        label="±1 std",
    )

    ax[i].set_ylabel("")
    ax[i].set_ylim(ymin, ymax)
    ax[i].spines["left"].set_position(("outward", 10))


ax[0].set_xlabel(f"vertical velocity / {ds['wvel'].units}")
ax[1].set_xlabel(f"moist static energy / {ds['h_mean'].units}")

ax[0].axvline(0, color="k", linestyle=":", alpha=0.5, linewidth=1)
ax[0].set_ylabel("z / m")
sns.despine()

plt.savefig(f"{PROJECT_ROOT}/figures/mse_dynamics.pdf", bbox_inches="tight")

# %%
