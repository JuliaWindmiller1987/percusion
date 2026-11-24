"""Central registry of dataset locations used in the PERCUSION analysis."""

import xarray as xr

# Fixed IPFS CIDs (not ipns://latest) so the figures stay reproducible
DATASETS = {
    # HALO
    "bahamas": "ipfs://bafybeias3h5uxtt4ky4d4gn6l6gxjqfkzbde5jlunya6g3umnkvn7xoyoe",
    "bahamas2": "ipfs://bafybeif52irmuurpb27cujwpqhtbg5w6maw4d7zppg2lqgpew25gs5eczm",  # used in flight_heights.py; rename as fits
    "dropsondes": "ipfs://bafybeibgeeqs5uhmbqy4hz4v3pihrfisiklcetisxkf63d2r473szaprwi",  # BEACH level 4
    "hamp_radar": "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a",
    "hamp_radiometer": "ipfs://bafybeihvsq73hokuex44oueqvnotaeoikwcs5yyiydtc777td2ldcm6z4y",
    "bacardi": "ipfs://bafybeic7yakug2r2dbxsk6b5oi7ajvztlxu3agk3dsaho6rc5weaod6qdy",
    # WALES
    "wales": "/work/mh0066/m301046/data/wales/wales.zarr",
    "wales_wv": "/work/mh0066/m301046/data/wales/waleswv.zarr",
    # Model
    "lam": "https://eerie.cloud.dkrz.de/datasets/orcestra_1250m_2d_hpz12/kerchunk",
}

# Extra arguments some datasets need
_OPEN_KWARGS = {
    "lam": {"chunks": {}, "zarr_format": 3},
}


def open_dataset(name, **kwargs):
    """Open a dataset by its short name, e.g. open_dataset("dropsondes")."""
    try:
        location = DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}"
        ) from None
    return xr.open_dataset(
        location, engine="zarr", **{**_OPEN_KWARGS.get(name, {}), **kwargs}
    )
