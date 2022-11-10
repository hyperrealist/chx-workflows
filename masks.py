import numpy as np
import tiled

from functools import reduce
from numpy import dtype
from pathlib import Path
from tiled.queries import Key

DETECTOR_SHAPES = {'eiger4m_single_image': (2167, 2070),
                   'eiger1m_single_image': (1065, 1030),
                   'eiger500K_single_image': (514, 1030)}


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


def combine_masks(masks):

        """
        Get a composite mask from tiled.
        This does not support version specification.
        This combines all masks in the list masks and returns a single mask.

        Parameters
        ----------
        masks: list
            list of masks

        Returns
        -------
        mask: DaskArray
        """

        return reduce(lambda x, y: x & y, masks)

class MaskClient:
    """
    MaskClient is a client for a tiled mask database.

    - Each registered Mask has a unique identifier (uid).
    - Arbitrary metadata can be registered along with the Mask.
    - We can search for Masks by detector, name, or other metadata.
    - A Mask is a NumPy array of Bools, that matches the detector shape,
      where False represents the values to be masked.
    - detector_name + mask_name + version is unique.
    - When you get a Mask from the registry it's type is a DaskArray,
      this is to support laziness and parallelization. To get the NumPy
      Array call compute() on it.
    """

    def __init__(self, tiled_client):
        self._tiled_client = tiled_client

    @classmethod
    def from_profile(cls, *args, **kwargs):
        return cls(tiled.client.from_profile(*args, **kwargs))

    def register_mask(self, detector, name, mask, version=0, optional_metadata={}):

        """
        Save a mask into tiled.

        Parameters
        ----------
        name: string
        detector: string
            The name of the detector that the mask is for.
        mask: numpy.ndarray
        version: integer, optional
            Verion information

        Returns
        -------
        node: tiled.Node

        """
        if not isinstance(mask, np.ndarray):
            raise ValueError("The mask must be a numpy.ndarray.")

        if mask.shape != DETECTOR_SHAPES[detector]:
            raise ValueError("Mask shape {mask.shape} does not match the detector"
                             " shape {DETECTOR_SHAPES[detector].")

        if mask.dtype != dtype('bool'):
            raise ValueError("Mask dtype {mask.dtype} must be dtype('bool').")

        metadata = {"spec": "mask",
                    "name": name,
                    'detector': detector,
                    'version': version}

        metadata.update(optional_metadata)

        # Make sure the mask doesn't already exist.
        masks = self._tiled_client.search(Key("spec") == "mask") \
                           .search(Key("name") == name) \
                           .search(Key("detector") == detector) \
                           .search(Key("version") == version)

        if len(masks):
            raise RuntimeError("A mask with this name and detector already exists.")

        result = self._tiled_client.write_array(
            mask,
            metadata=metadata,
        )

        return result

    def get_mask(self, detector, name, version=None):

        """
        Get a mask from tiled.

        Parameters
        ----------
        detector: string
        name: string
        version: integer, None
            A None value here will return the highest version.

        Returns
        -------
        mask_uid: str
        mask: DaskArray
        """

        results = self._tiled_client.search(Key("spec") == "mask") \
                                    .search(Key("name") == name) \
                                    .search(Key('detector') == detector)

        if version is None:
            uid, mask = max(results.items(), 
                            key=lambda item: item[1].metadata['version'])
        else:
            results = results.search(Key('version') == version)
            assert len(results) == 1
            uid, mask = list(results.items())[0]

        return uid, mask.read()

    def get_mask_by_uid(self, uid):

        """
        Get a mask from tiled.
nod
        Parameters
        ----------
        uid: string

        Returns
        -------
        mask: DaskArray
        """

        return self._tiled_client[uid].read()

    def get_mask_uid(self, detector, name, version):

        """
        Get a mask_uid from tiled.

        Parameters
        ----------
        detector: string
        name: string
        version: int

        Returns
        -------
        uid: string
        """

        results = self._tiled_client.search(Key("spec") == "mask") \
                                    .search(Key("name") == name) \
                                    .search(Key('detector') == detector) \
                                    .search(Key('version') == version)
        assert len(results) == 1
        return list(results)[0]

    def delete_mask(self, detector, name, version):

        """
        Delete a mask from tiled.

        Parameters
        ----------
        detector: string
        name: string
        version: int
        """

        results = self._tiled_client.search(Key("spec") == "mask") \
                                    .search(Key("name") == name) \
                                    .search(Key('detector') == detector) \
                                    .search(Key('version') == version)

        uids = list(results)
        for uid in uids:
            del self._tiled_client[uid]

    def list_masks(self):

        """
        Get a list of the available masks.

        Returns
        -------
        mask_details: list
        """

        results = self._tiled_client.search(Key("spec") == "mask")
        return [(node.metadata.get('detector', 'any'),
                 node.metadata["name"],
                 node.metadata.get("version"))
                for node in results.values()]
