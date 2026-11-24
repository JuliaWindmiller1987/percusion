# %%
"""Lower-tropospheric buoyancy B_L (Ahmed et al. 2020) from BEACH dropsondes.

B_L = CAPE_L - SUBSAT_L, expressed in K (multiplied by THETA_E0 / g):
  CAPE_L   = g * w_pbl * (theta_e,pbl - theta_e*,lft) / theta_e*,lft
  SUBSAT_L = g * w_lft * (theta_e*,lft - theta_e,lft) / theta_e*,lft
pbl: surface to 150 hPa above the surface; lft: top of pbl to 500 hPa.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from metpy.units import units

from pathlib import Path
import percusion
from percusion.datasets import open_dataset

PROJECT_ROOT = Path(percusion.__file__).resolve().parents[2]

PERIODS = {
    "East Atlantic": ("2024-08-11", "2024-09-05"),
    "West Atlantic": ("2024-09-07", "2024-09-28"),
    "all": ("2024-08-11", "2024-09-28"),
}

G = 9.81  # m s-2


# %% --- buoyancy ---------------------------------------------------------------


def layer_weights(dp_pbl, dp_lft):
    """Deep-inflow layer weights (Ahmed et al. 2020, with a = b)."""
    w_pbl = dp_pbl / dp_lft * np.log((dp_pbl + dp_lft) / dp_pbl)
    return w_pbl, 1.0 - w_pbl


def buoyancy_profile(p_hpa, ta_k, q_kgkg, dp_pbl=150.0, p_top=500.0, dp=5.0):
    """B_L, CAPE_L and SUBSAT_L (all in K) for one profile.

    Returns NaNs if the profile does not reach 500 hPa or does not get close
    to the surface (990 hPa).
    """
    valid = np.isfinite(p_hpa) & np.isfinite(ta_k) & np.isfinite(q_kgkg)
    if valid.sum() < 10:
        return np.nan, np.nan, np.nan

    # sort by pressure (np.interp needs increasing x)
    order = np.argsort(p_hpa[valid])
    p = p_hpa[valid][order]
    t = ta_k[valid][order]
    q = q_kgkg[valid][order]

    p_sfc = p.max()  # lowest measured level as surface pressure
    if p.min() > p_top or p_sfc < 990:
        return np.nan, np.nan, np.nan

    # regular pressure grid -> simple means are mass-weighted means
    p_grid = np.arange(p_top, p_sfc + 1e-6, dp)
    P = p_grid * units.hPa
    T = np.interp(p_grid, p, t) * units.K
    Q = np.interp(p_grid, p, q) * units("kg/kg")

    Td = mpcalc.dewpoint_from_specific_humidity(P, T, Q)
    theta_e = mpcalc.equivalent_potential_temperature(P, T, Td).m_as("K")
    theta_es = mpcalc.saturation_equivalent_potential_temperature(P, T).m_as("K")

    p_pbl_top = p_sfc - dp_pbl
    in_pbl = p_grid >= p_pbl_top
    in_lft = ~in_pbl

    theta_e_pbl = theta_e[in_pbl].mean()
    theta_e_lft = theta_e[in_lft].mean()
    theta_es_lft = theta_es[in_lft].mean()

    w_pbl, w_lft = layer_weights(dp_pbl, p_pbl_top - p_top)

    cape_l = G * w_pbl * (theta_e_pbl - theta_es_lft) / theta_es_lft
    subsat_l = G * w_lft * (theta_es_lft - theta_e_lft) / theta_es_lft
    return cape_l - subsat_l, cape_l, subsat_l


def add_buoyancy(ds, sonde_dim="sonde"):
    """Add B_L, CAPE_L and SUBSAT_L (one value per sonde) to the dataset."""
    results = np.array(
        [
            buoyancy_profile(
                ds["p"].isel({sonde_dim: i}).values / 100.0,  # Pa -> hPa
                ds["ta"].isel({sonde_dim: i}).values,
                ds["q"].isel({sonde_dim: i}).values,
            )
            for i in range(ds.sizes[sonde_dim])
        ]
    )
    for k, name in enumerate(["B_L", "CAPE_L", "SUBSAT_L"]):
        ds[name] = (sonde_dim, results[:, k])
        ds[name].attrs["units"] = "m s-2"
    return ds


# %% --- load data and compute buoyancy --------------------------------------------

sondes = open_dataset("dropsondes")
sondes = add_buoyancy(sondes)

# %% --- scatter plot: B_L vs CWV, coloured by launch latitude --------------------

fig, axes = plt.subplots(
    1, 2, figsize=(10, 2.75), sharey=True, sharex=True, constrained_layout=True
)

for ax, (name, (start, end)) in zip(axes, PERIODS.items()):
    s = sondes.sel(launch_time=slice(start, end))
    sc = ax.scatter(
        s.iwv, s.B_L, c=s.launch_lat, s=7, vmin=5, vmax=15, clip_on=False, marker="o"
    )

    # ax.set_title(name)
    ax.set_xlabel("CWV / mm")
    ax.set_xticks(np.arange(35, 71, 5))
    ax.set_xlim(32, 70)
    ax.set_ylim(-0.7, 0.15)

    ax.text(
        -0.18 if name == "East Atlantic" else -0.11,
        1.05,
        f"{chr(97 + list(PERIODS.keys()).index(name))})",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )

    ax.text(
        52,
        0.1,
        name,
        fontsize=9,
        va="bottom",
        ha="center",
    )


axes[0].set_ylabel("B$_L$ / m s$^{-2}$")
sns.despine(offset=5)

fig.colorbar(sc, ax=axes, label="launch latitude / °N", pad=0.01)

plt.savefig(
    f"{PROJECT_ROOT}/figures/sondes_buoyancy_vs_cwv.pdf",
    bbox_inches="tight",
)

# %%

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True, constrained_layout=True)

for ax, (name, (start, end)) in zip(axes, PERIODS.items()):
    s = sondes.sel(launch_time=slice(start, end))
    sc = ax.scatter(s.launch_lat, s.iwv, c=s.B_L, s=15, vmin=-0.7, vmax=0.2)
    ax.set_title(name)
    ax.set_xlabel("lat / °N")
    ax.set_ylim(32, 70)

axes[0].set_ylabel("CWV / mm")
sns.despine(offset=5)
fig.colorbar(sc, ax=axes, label="B$_L$ / m s$^{-2}$", pad=0.01)

# %%
