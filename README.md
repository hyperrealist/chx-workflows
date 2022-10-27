# CHX Workflows

## Using Prefect for pre-processing


## Mask Registry
- The Mask registry is a tiled database for masks.
- Each registered Mask has a unique identifier (uid).
- Arbitrary metadata can be registered along with the Mask.
- We can search for Masks by detector, name, or other metadata.
- A Mask is a NumPy array of Bools, that matches the detector shape, where False represents the values to be masked.
- detector_name + name is unique
- When you get a Mask from the registry it's type is a DaskArray, this is to support laziness and parallelization.  To get the NumPy Array call `compute()` on it.


**Register a Mask**


    from masks import register_mask
    
    mask = array([[False, False, False, ..., False, False, False],
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           ...,
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           [False, False, False, ..., False, False, False]])
    
    register_mask('eiger4m_single_image', 'chip_mask', mask)


**List available Masks**


    from masks import get_masks
    get_masks()


    In [12]: get_masks()
    Out[12]:
    [('eiger4m_single_image', 'pixel_mask'),
     ('eiger4m_single_image', 'bad_pixels'),
     ('eiger4m_single_image', 'chip_mask'),
     ('eiger4m_single_image', 'jul11_2022_4m_saxs'),
     ('eiger500K_single_image', 'chip_mask'),
     ('eiger1m_single_image', 'chip_mask')]


**Get a Mask**


    from masks import get_mask
    mask = get_mask('eiger4m_single_image', 'chip_mask').compute()


    In [8]: mask
    Out[8]:
    array([[False, False, False, ..., False, False, False],
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           ...,
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           [False, False, False, ..., False, False, False]])


**Get a Composite Mask**


    from masks import get_composite_mask
    mask_names = ["pixel_mask", "chip_mask", "bad_pixels", "jul11_2022_4m_saxs"]
    mask = get_composite_mask('eiger4m_single_image', mask_names)


    In [7]: mask
    Out[7]: dask.array<and_, shape=(2167, 2070), dtype=bool, chunksize=(2167, 2070), chunktype=numpy.ndarray>
    
    In [8]: mask.compute()
    Out[8]:
    array([[False, False, False, ..., False, False, False],
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           ...,
           [False,  True,  True, ...,  True,  True, False],
           [False,  True,  True, ...,  True,  True, False],
           [False, False, False, ..., False, False, False]])


**Get the uid of a Mask**


    from masks import get_mask_uid
    get_mask_uid('eiger4m_single_image', 'chip_mask')
    Out[9]: '73f07be0-d535-41db-9258-2dd01e1ac37b'

**Delete Masks**


    from masks import delete_mask
    delete_mask('eiger4m_single_image', 'chip_mask')


## Testing



