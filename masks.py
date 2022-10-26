import numpy as np
import tiled

from functools import reduce
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


def register_mask(detector, name, mask, optional_metadata={}):

    """
    Save a mask into tiled.

    Parameters
    ----------
    name: string
    detector: string
        The name of the detector that the mask is for.
    mask: array

    Returns
    -------
    node: tiled.Node

    """

    metadata = {"spec": "mask", "name": name, 'detector': detector}
    metadata.update(optional_metadata)

    # Make sure the mask doesn't already exist.
    masks = mask_client.search(Key("spec") == "mask") \
                       .search(Key("name") == name) \
                       .search(Key("detector") == detector)
    if len(masks):
        raise RuntimeError("A mask with this name and detector already exists.")

    result = mask_client.write_array(
        mask,
        metadata=metadata,
    )

    return result


def get_mask(detector, name):

    """
    Get a mask from tiled.

    Parameters
    ----------
    detector: string
    name: string

    Returns
    -------
    mask: DaskArray
    """

    results = mask_client.search(Key("spec") == "mask") \
                         .search(Key("name") == name) \
                         .search(Key('detector') == detector)
    assert len(results) == 1
    return results.values().first().read()


def get_combined_mask(detector, mask_names):

    """
    Get a mask from tiled.

    Parameters
    ----------
    detector: string
    mask_names: list
        list of mask_names

    Returns
    -------
    mask: DaskArray
    """

    masks = [get_mask(detector, mask) for mask in mask_names]
    return reduce(lambda x, y: x & y, masks)


def get_mask_uid(detector, name):

    """
    Get a mask_uid from tiled.

    Parameters
    ----------
    detector: string
    name: string

    Returns
    -------
    uid: string
    """

    results = mask_client.search(Key("spec") == "mask") \
                         .search(Key("name") == name) \
                         .search(Key('detector') == detector)
    assert len(results) == 1
    return list(results)[0]


def delete_masks(detector, name):

    """
    Delete a mask from tiled.

    Parameters
    ----------
    detector: string
    name: string
    """

    results = mask_client.search(Key("spec") == "mask") \
                         .search(Key("name") == name) \
                         .search(Key('detector') == detector)
    uids = list(results)
    for uid in uids:
        del mask_client[uid]


def get_masks():

    """
    Get a list of the available masks.

    Returns
    -------
    mask_names: list
    """

    results = mask_client.search(Key("spec") == "mask")
    return [(node.metadata.get('detector', 'any'), node.metadata["name"]) 
            for node in results.values()]
