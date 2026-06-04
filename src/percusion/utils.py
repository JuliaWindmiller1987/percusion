import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import warnings

lon_min = -62
lon_max = -10
lat_min = -2
lat_max = 22

campaign_start, campaign_end = "2024-08-11 11:59:00", "2024-09-28 20:02:00"


def base_map(
    lon_min=lon_min,
    lon_max=lon_max,
    lat_min=lat_min,
    lat_max=lat_max,
    size=8,
    aspect=16 / 9,
    coastline_kwargs=None,
    ax=None,
):
    map_extent = [lon_min, lon_max, lat_min, lat_max]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(size * aspect, size),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = ax.figure

        if not isinstance(ax.projection, ccrs.PlateCarree):
            warnings.warn(
                f"Expected a PlateCarree projection, got {ax.projection!r}",
                UserWarning,
            )

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    default_coast = {"color": "white", "alpha": 1.0}
    default_coast.update(coastline_kwargs)
    ax.coastlines(**default_coast)

    gl = ax.gridlines(
        draw_labels=True,
        alpha=0.0,
        xlocs=range(-60, -10, 10),
        ylocs=range(0, 25, 5),
    )

    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def list_of_kinds(list_of_dicts):

    lists_of_kinds = [
        dict["kinds"] if isinstance(dict["kinds"], (list)) else [dict["kinds"]]
        for dict in list_of_dicts
    ]
    kinds = [i for list_of_kinds in lists_of_kinds for i in list_of_kinds]

    return list(set(kinds))


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


def kinds2color(kinds):
    if "circle" and "atr_coordination" in kinds:
        return "#e9c46a"
    elif "circle" in kinds:
        return "#2ec4b6"
    elif "ec_track" in kinds:
        return "#f4a261"
    return "#e76f51"
