import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_hamp_radiometer_data():
    ds = xr.open_dataset("ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y", engine="zarr")
    return ds

def get_stereo_data(data_path="/scratch/l/L.Volkmer/cloud_geometry/percusion/PERCUSION_HALO_specMACS-POL_cloud_geometry.zarr"):
    ds = xr.open_dataset(data_path, engine="zarr")
    return ds

def main():

    ds_hamp = get_hamp_radiometer_data()
    print(ds_hamp)
    ds_stereo = get_stereo_data()
    print(ds_stereo)

    stereo_time = ds_stereo["time"]
    stereo_height = ds_stereo["height"]
    IWV = ds_hamp["IWV"]
    cwv_xmin, cwv_xmax = float(IWV.quantile(0.1).values), float(IWV.quantile(0.9).values)
    cwv_xmin, cwv_xmax = np.round([cwv_xmin, cwv_xmax], 0)
    IWV_bins = np.arange(cwv_xmin, cwv_xmax + 1, 1.0)
    IWV = IWV.interp(time=stereo_time.values)

    valid = ~np.isnan(IWV.values) & ~np.isnan(stereo_height.values)
    height_bins = np.arange(0, 14000, 10)

    print("Calc histogram")
    hist2d, xedges, yedges = np.histogram2d(IWV.values[valid], stereo_height.values[valid], bins=(IWV_bins, height_bins))
    xcoords = (xedges[1:] + xedges[:-1])/2
    ycoords = (yedges[1:] + yedges[:-1])/2
    ntot = len(stereo_height.values[valid])
    
    paper_tobi = "Kölling, T., Zinner, T., and Mayer, B.: Aircraft-based stereographic reconstruction of 3-D cloud geometry, Atmos. Meas. Tech., 12, 1155–1166, https://doi.org/10.5194/amt-12-1155-2019, 2019."
    paper_lea = "Volkmer, L., Kölling, T., Zinner, T., and Mayer, B.: Consideration of the cloud motion for aircraft-based stereographically derived cloud geometry and cloud top heights, Atmos. Meas. Tech., 17, 6807–6817, https://doi.org/10.5194/amt-17-6807-2024, 2024."
    hamp_data = "Boemeke, J.. (2025). Radiometer Data from the Halo Microwave Package (Level 2). ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y"
    ds_out = xr.Dataset({
        "histogram": xr.DataArray(data=hist2d, dims=("cwv", "height"), coords={"cwv": xcoords, "height": ycoords}, attrs={"units": "count", "long_name": "stereo_height_in_cwv_space_histogram"}),
        "ntot": xr.DataArray(ntot, dims=(), attrs={"long_name": "total number of stereo points"}),
    }, 
    )
    ds_out.attrs["description"] = "Histogram of stereographically derived points from specMACS (downward-looking) RGB cameras binned to column water vapor from HAMP radiometers."
    ds_out.attrs['references'] = paper_tobi + " " + paper_lea + " " + hamp_data
    ds_out.attrs['contact'] = "Lea Volkmer, L.Volkmer@physik.uni-muenchen.de"
    ds_out.attrs['institution'] = "Meteorological Institut, Ludwig-Maximilians-University Munich"
    ds_out.to_netcdf("stereo_height_vs_CWV_hist.nc")

    #hist2d /= ntot

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #cf = ax.contourf(xcoords, ycoords, np.where(hist2d.T > 0.00018, hist2d.T, np.nan), cmap="plasma", levels=np.arange(0.0002, 0.0065, 0.0004), extend='both')
    #cf = ax.contourf(xcoords, ycoords, np.where(hist2d.T > 1e-7, hist2d.T, np.nan), cmap="plasma", levels=np.geomspace(1e-7, 0.1, 60), norm="log", extend='both')
    cf = ax.contourf(xcoords, ycoords, np.where(hist2d.T > 0, hist2d.T, np.nan), cmap="plasma", norm="log", extend='both', levels=20)
    ax.set_xlabel("CWV / mm")
    ax.set_ylabel("height / m")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(cf, cax = cax, label = "frequency")
    plt.tight_layout()
    outname = "stereo_height_vs_CWV.png"
    plt.savefig(outname, bbox_inches="tight", dpi=300, transparent=True)
    print(f"Saved figure to {outname}")

if __name__ == "__main__":
    main()
