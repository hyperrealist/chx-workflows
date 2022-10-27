import numpy as np
import tiled

from functools import reduce
from pathlib import Path
from tiled.queries import Key


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


class MaskClient:
    """
    MaskClient is a client for a tiled mask database.
    
    - Each registered Mask has a unique identifier (uid).
    - Arbitrary metadata can be registered along with the Mask.
    - We can search for Masks by detector, name, or other metadata.
    - A Mask is a NumPy array of Bools, that matches the detector shape, 
      where False represents the values to be masked. 
    - detector_name + name is unique.  
    - When you get a Mask from the registry it's type is a DaskArray, 
      this is to support laziness and parallelization. To get the NumPy 
      Array call compute() on it.
    """
    
    def __init__(self, tiled_client):
        self._tiled_client = tiled_client

    @classmethod
    def from_profile(cls, *args, **kwargs):
        return cls(tiled.client.from_profile(*args, **kwargs))

    def register_mask(self, detector, name, mask, optional_metadata={}):

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
        masks = self._tiled_client.search(Key("spec") == "mask") \
                           .search(Key("name") == name) \
                           .search(Key("detector") == detector)
        if len(masks):
            raise RuntimeError("A mask with this name and detector already exists.")

        result = self._tiled_client.write_array(
            mask,
            metadata=metadata,
        )

        return result


    def get_mask(self, detector, name):

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

        results = self._tiled_client.search(Key("spec") == "mask") \
                                    .search(Key("name") == name) \
                                    .search(Key('detector') == detector)
        assert len(results) == 1
        return results.values().first().read()


    def get_composite_mask(self, detector, mask_names):

        """
        Get a composite mask from tiled.
        This combines all masks in the list mask_names and returns a single mask.

        Parameters
        ----------
        detector: string
        mask_names: list
            list of mask_names

        Returns
        -------
        mask: DaskArray
        """

        masks = [self.get_mask(detector, mask) for mask in mask_names]
        return reduce(lambda x, y: x & y, masks)


    def get_mask_uid(self, detector, name):

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

        results = self._tiled_client.search(Key("spec") == "mask") \
                                    .search(Key("name") == name) \
                                    .search(Key('detector') == detector)
        assert len(results) == 1
        return list(results)[0]


    def delete_mask(self, detector, name):

        """
        Delete a mask from tiled.

        Parameters
        ----------
        detector: string
        name: string
        """

        results = self._tiled_client.search(Key("spec") == "mask") \
                                    .search(Key("name") == name) \
                                    .search(Key('detector') == detector)
        uids = list(results)
        for uid in uids:
            del self._tiled_client[uid]


    def get_masks(self):

        """
        Get a list of the available masks.

        Returns
        -------
        mask_names: list
        """

        results = self._tiled_client.search(Key("spec") == "mask")
        return [(node.metadata.get('detector', 'any'), node.metadata["name"]) 
                for node in results.values()]
