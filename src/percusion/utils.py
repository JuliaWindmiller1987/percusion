import matplotlib.pyplot as plt
import cartopy.crs as ccrs

lon_min = -62
lon_max = -10
lat_min = -2
lat_max = 22


def base_map(
    lon_min=lon_min,
    lon_max=lon_max,
    lat_min=lat_min,
    lat_max=lat_max,
    size=8,
    aspect=16 / 9,
    coastline_kwargs=None,
):
    map_extent = [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(
        figsize=(size * aspect, size),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    default_coast = {"color": "white", "alpha": 1.0}
    default_coast.update(coastline_kwargs)
    ax.coastlines(**default_coast)

    gl = ax.gridlines(
        draw_labels=True,
        alpha=0.25,
        xlocs=range(-60, -10, 10),
        ylocs=range(0, 25, 5),
    )

    gl.top_labels = False
    gl.right_labels = False
    return fig, ax
