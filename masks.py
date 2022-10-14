import numpy as np
import tiled

from pathlib import Path
from tiled.client import from_profile
from tiled.queries import Key

mask_client = from_profile("nsls2", "dask", username=None)["chx"]["sandbox"]


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


def register_mask(mask, name, optional_metadata={}):

    """
    Save a mask into tiled.

    Parameters
    ----------
    name: string
    mask: array

    Returns
    -------
    node: tiled.Node

    """

    metadata = {"description": "mask", "name": name}
    metadata.update(optional_metadata)

    masks = mask_client.search(Key("spec") == "mask").search(Key("name") == name)
    if len(masks):
        raise RuntimeError("A mask with this name already exists.")

    result = mask_client.write_array(
        mask,
        metadata=metadata,
    )

    return result


def get_mask(name):

    """
    Get a mask from tiled.

    Parameters
    ----------
    name: string

    Returns
    -------
    mask: DaskArray
    """

    results = mask_client.search(Key("spec") == "mask").search(Key("name") == name)
    assert len(results) == 1

    return results.values().first().read()


def delete_masks(name):

    """
    Delete a mask from tiled.

    Parameters
    ----------
    name: string
    """

    results = mask_client.search(Key("spec") == "mask").search(Key("name") == name)
    uids = list(results)
    for uid in uids:
        del mask_client[uid]


def get_mask_names():

    """
    Get a list of the available masks.

    Returns
    -------
    mask_names: list
    """

    results = mask_client.search(Key("spec") == "mask")
    return [node.metadata["name"] for node in results.values()]
