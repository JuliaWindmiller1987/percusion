# %%
import numpy as np
import matplotlib.pyplot as plt
from percusion.datasets import open_dataset
from percusion import utils

campaign_start = utils.campaign_start
campaign_end = utils.campaign_end

# %%
m_to_feet = 3.28084

# %%

ds_orcestra = (
    utils.get_halo_position()
    .sel(time=slice(campaign_start, campaign_end))
    .dropna(dim="time", subset=["pressure_altitude_m"])
)

ds_orcestra["flight_level"] = ds_orcestra.pressure_altitude_m * m_to_feet * 1e-2

# %%

flight_levels = {
    "ATR circle": 355,
    "HALO initial phase": 410,  # actually planned was typically 410
    "HALO mid phase": 430,
    "HALO final phase": 450,
}

delta_flight_levels = 10
# %%
plt.figure(figsize=(6, 4))
ds_orcestra["flight_level"].plot.hist(bins=np.arange(320, 500, 1), density=True)

for name, level in flight_levels.items():
    plt.axvline(level, color="C1", linestyle="-", label=name)
    plt.axvline(level - delta_flight_levels, color="C1", linestyle=":", alpha=0.5)
    plt.axvline(level + delta_flight_levels, color="C1", linestyle=":", alpha=0.5)

plt.xlabel("Flight level / ft")
plt.legend()

plt.figure(figsize=(6, 4))
ds_orcestra["pressure_altitude_m"].plot.hist(
    bins=np.arange(10e3, 15e3, 100), density=True
)

for name, level in flight_levels.items():
    level_m = level / (m_to_feet * 1e-2)
    delta_m = delta_flight_levels / (m_to_feet * 1e-2)
    plt.axvline(level_m, color="C1", linestyle="-")
    plt.axvline(level_m + delta_m, color="C1", linestyle=":", alpha=0.5)
    plt.axvline(level_m - delta_m, color="C1", linestyle=":", alpha=0.5)
plt.xlabel("Flight level / m")
plt.legend()

# %%

num_points = ds_orcestra["flight_level"].size


all_spec_fl_points = 0

for name, level in flight_levels.items():

    cond = (ds_orcestra["flight_level"] > level - delta_flight_levels) & (
        ds_orcestra["flight_level"] <= level + delta_flight_levels
    )

    num_fl_points = (
        ds_orcestra["flight_level"]
        .where(
            cond,
            drop=True,
        )
        .size
    )

    mean_hp = (
        ds_orcestra["pressure_altitude_m"]
        .where(
            cond,
            drop=True,
        )
        .mean()
        .values
    )

    all_spec_fl_points += num_fl_points

    print(
        f"{name}: {num_fl_points} points ({num_fl_points/num_points*100:.1f} % of total points) at flight level {level} ft, mean pressure height: {mean_hp:.1f} m"
    )

print(
    f"Total points in specified flight levels: {all_spec_fl_points} ({all_spec_fl_points/num_points*100:.1f} % of total points)"
)

print(
    f"Median height of all points: {ds_orcestra['pressure_altitude_m'].median().values:.1f} m"
)

print(
    f"Maximum height of all points: {ds_orcestra['pressure_altitude_m'].max().values:.1f} m"
)
# %%
