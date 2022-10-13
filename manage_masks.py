import numpy as np
import tiled

from pathlib import Path
from tiled.client import from_profile

MASK_DIR = Path("/nsls2/data/chx/legacy/analysis/masks")

tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
tiled_client_sandbox = tiled_client["sandbox"]


def load_array_from_file(filename):

    """
    This function reads in pixel mask files and
    outputs a 1d or 2d numpy array object
    """
    return np.load(filename)


def list_to_mask(pixel_list, shape=(2167, 2070)):

    """
    This function accepts a 1d array and returns a
    2d mask object
    """

    pixel_mask = np.ones(shape=shape)
    pixel_mask.ravel()[pixel_list] = 0

    return pixel_mask.astype(bool)


def register_mask(mask, name, metadata={}):

    """
    This function saves a mask into tiled
    """

    md = {'spec': 'mask', 'name': name}
    md.update(metadata)

    result = tiled_client_sandbox.write_array(
            mask,
            metadata=md,
        )

    return result.item["id"]


def get_mask(name):

    """
    This function returns the mask "name" from
    tiled
    """

    results = tiled_client_sandbox.search(Key('spec')=='mask').search(Key('name')==name)
    assert len(results) == 1

    return results.values().first().read()
