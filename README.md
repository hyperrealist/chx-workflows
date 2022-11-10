# CHX Workflows

## Mask Registry
- The Mask registry is a tiled database for masks.
- Each registered Mask has a unique identifier (uid).
- Arbitrary metadata can be registered along with the Mask.
- We can search for Masks by detector, name, or other metadata.
- A Mask is a NumPy array of Bools, that matches the detector shape, where False represents the values to be masked.
- detector_name + mask_name + version is unique
- When you get a Mask from the registry its type is a DaskArray, this is to support laziness and parallelization. To get the NumPy Array call `compute()` on it.

**Create a MaskClient**


    from masks import MaskClient
    
    sandbox = from_profile("nsls2", "dask", username=None)\["chx"\]["sandbox"]
    mask_clilent = MaskClient(sandbox)


**List available Masks**
It prints the detector_name, mask_name, and version.


    In [12]: mask_client.list_masks()
    Out[12]:
    [('eiger4m_single_image', 'pixel_mask', 0),
     ('eiger4m_single_image', 'bad_pixels', 0),
     ('eiger4m_single_image', 'chip_mask', 0),
     ('eiger4m_single_image', 'jul11_2022_4m_saxs', 0),
     ('eiger500K_single_image', 'chip_mask', 0),
     ('eiger1m_single_image', 'chip_mask', 0)]
    
    

**Register a Mask**

- Version is optional and defaults to 0.
- detector_name + mask_name + version must be unique.
- The mask shape must match the detector shape
- The array dtype must be bool


    mask = array([[False, False, False, ..., False, False, False],
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           ...,
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           [False, False, False, ..., False, False, False]])
    
    # Detector name, mask_name, mask
    mask_client.register_mask('eiger4m_single_image', 'chip_mask', mask)


**Get a Mask**
Version is optional and defaults to returning the highest version number mask.
Get mask returns a uid, along with the mask.  Without specifying a version, the mask that get_mask returns might not always be the same.  So we have to return a uid, so that we know exactly which mask we have.


    uid, mask = mask_client.get_mask('eiger4m_single_image', 'chip_mask')


    In [8]: mask.compute()
    Out[8]:
    array([[False, False, False, ..., False, False, False],
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           ...,
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           [False, False, False, ..., False, False, False]])


**Combine a list of masks.**


    from masks import combine_masks
    mask1 = mask_client.get_mask('eiger4m_single_image', 'chip_mask', version=0)
    mask2 = mask_client.get_mask('eiger4m_single_image', 'bad_pixels')
    masks = [mask1, mask2]
    combined_mask = combine_masks(masks).compute()


    In [64]: combined_mask
    Out[64]:
    array([[False, False, False, ..., False, False, False],
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           ...,
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           [False, False, False, ..., False, False, False]])

**Get the uid of a Mask**


    mask_client.get_mask_uid('eiger4m_single_image', 'chip_mask', 0)
    Out[9]: '73f07be0-d535-41db-9258-2dd01e1ac37b'

**Delete Masks**


    mask_client.delete_mask('eiger4m_single_image', 'chip_mask', 0)


## Testing

There are several tests written in the test_sparsify.py.
You can run the tests by typing `pytest`
