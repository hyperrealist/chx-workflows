import databroker
import distributed
import event_model
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import prefect
import sparse
import tiled
import time

from collections import defaultdict
from pathlib import Path
from pandas import Timestamp
from prefect import Flow, Parameter, task
from tiled.client import from_profile
from tiled.structures.sparse import COOStructure

EXPORT_PATH = Path("/nsls2/data/dssi/scratch/prefect-outputs/chx/")
MASK_DIR = Path("/nsls2/data/chx/legacy/analysis/masks")

# distributed_client = distributed.Client(n_workers=1, threads_per_worker=1, processes=False)
tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
tiled_client_chx = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]

run1 = tiled_client_chx["d85d157f-57d9-4649-9b65-0d3b9f754e01"]
run2 = tiled_client_chx["e909f4a2-12e3-4521-a7a6-be2b728a826b"]
run3 = tiled_client_chx["b79184e1-d053-42e4-b1eb-f8ab0a146220"]

run2_file_export = "/nsls2/data/dssi/scratch/prefect-outputs/chx/compressed_data/uid_e909f4a2-12e3-4521-a7a6-be2b728a826b.cmp"
run2_file_chx = "/nsls2/data/chx/legacy/Compression_test/fluerasu/uid_e909f4a2-12e3-4521-a7a6-be2b728a826b.cmp"


def get_pixel_mask(metadata):
    return (1 - np.array(metadata["pixel_mask"], dtype=bool)).astype(bool)


def get_bad_pixel_mask(metadata):
    bad_pixel_files = {
        "eiger4m_single_image": MASK_DIR / "BadPix_4M.npy",
        "image": MASK_DIR / "BadPix_4M.npy",
    }

    if bad_pixel_files.get(metadata["detector"]):
        bad_pixel_list = np.load(bad_pixel_files[metadata["detector"]])
    else:
        bad_pixel_list = np.array([], dtype=bool)

    bad_pixel_mask = np.ones(shape=(2167, 2070))
    bad_pixel_mask.ravel()[bad_pixel_list] = 0

    return bad_pixel_mask.astype(bool)


def get_chip_mask(metadata):
    chip_mask_files = {
        "eiger1m_single_image": MASK_DIR / "Eiger1M_Chip_Mask.npy",
        "eiger4m_single_image": MASK_DIR / "Eiger4M_chip_mask.npy",
        "image": MASK_DIR / "Eiger4M_chip_mask.npy",
        "eiger500K_single_image": MASK_DIR / "Eiger500K_Chip_Mask.npy",
    }

    # Load the chip mask.
    assert chip_mask_files.get(metadata["detector"])
    chip_mask = np.load(chip_mask_files.get(metadata["detector"]))

    return chip_mask.astype(bool)


def get_custom_mask(
    filename="/nsls2/data/chx/legacy/analysis/2022_2/masks/Jul11_2022_4M_SAXS.npy",
):
    mask = np.load(filename)
    mask = np.flip(mask, axis=0)
    return mask


def get_file_metadata(run, detector="eiger4m_single_image"):
    dataset = run[f"{detector}_metadata_patched_in_at_runtime"]["data"].read()
    file_metadata = {key: dataset[key].values[0] for key in list(dataset)}

    # Convert numpy arrays to lists.
    for key in {"pixel_mask", "binary_mask"}:
        file_metadata[key] = file_metadata[key].tolist()

    return file_metadata


def get_run_metadata(run):
    """
    Collect the BlueskyRun metadata.

    Parameters
    ----------
    run: a BlueskyRun

    Returns
    -------
    metadata: dict
        The BlueskyRun's metadata
    """
    # TODO: Exception or Warning for collisions with the start metadata.

    metadata = {}
    metadata.update(run.start)
    metadata["suid"] = run.start["uid"].split("-")[0]  # short uid
    metadata.update(run.start.get('plan_args', {}))

    # Get the device metadata.
    detector = run.start['detectors'][0]
    metadata['detector'] = f'{detector}_image'
    # Check if the method below applies to runs in general and not just for run2
    metadata.update(run['primary'].descriptors[0]['configuration'][detector]['data'])


    # Get filename prefix.
    # We think we can still sparsify a run if there are no resources,
    # so we don't raise an exception if no resource is found.
    for name, document in run.documents():
        if name == 'resource':
            metadata['filename'] = Path(document.get("root", "/"), document["resource_path"])
            break

    if "primary" in run:
        descriptor = run["primary"].descriptors[0]
        # data_keys is a required key in the descriptor document.
        # detector_name must be in the descriptor.data_keys
        metadata["img_shape"] = descriptor["data_keys"][f'{detector}_image'].get('shape', [])[:2][::-1]

    # Fix up some datatypes.
    metadata["number of images"] = int(metadata["number of images"])
    metadata["start_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(run.start["time"])
    )
    metadata["stop_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(run.stop["time"])
    )

    return metadata


def write_sparse_chunk(data, dataset_id=None, block_info=None, 
                       dataset=None):
    result = sparse.COO(data)

    if block_info:
        if dataset is None:
            tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
            tiled_client_sandbox = tiled_client["sandbox"]
            dataset = tiled_client_sandbox[dataset_id]

        dataset.write_block(
                coords=result.coords,
                data=result.data,
                block=block_info[None]['chunk-location'],
            )
    return data


#@task
def sparsify(ref):
    """
    Performs sparsification.

    Parameters
    ----------
    ref: string
        This is the reference to the BlueskyRun to be exported. It can be
        a partial uid, a full uid, a scan_id, or an index (e.g. -1).

    """
    # Get the BlueskyRun from Tiled.
    run = tiled_client_chx[ref]

    # Compose the run metadata.
    metadata = get_run_metadata(run)
    metadata.update(get_file_metadata(run))

    # Load the images.
    dask_images = run["primary"]["data"]["eiger4m_single_image"].read()

    # Rotate the images if he detector is eiger500k_single_image.
    if 'eiger500K_single_image' == metadata["detector"]:
        dask_images = np.rotate(dask_images, axis=(3, 2))

    # Get the masks.
    pixel_mask = get_pixel_mask(metadata)
    chip_mask = get_chip_mask(metadata)
    bad_pixel_mask = get_bad_pixel_mask(metadata)
    custom_mask = get_custom_mask() # TODO: custom_mask should happen after the sparsification.
    final_mask = pixel_mask & chip_mask & bad_pixel_mask & custom_mask

    # Make the final mask the same shape as the images by extending the mask into a 3d array.
    num_images = dask_images.shape[1]
    mask3d = np.broadcast_to(final_mask, (num_images,) + final_mask.shape)

    # Flip the images to match mask orientation (TODO: probably better to flip the mask for faster computing).
    flipped_images = np.flip(dask_images, axis=2)

    # Apply the mask to flipped images.
    masked_images = flipped_images * mask3d
    chunksize = list(masked_images.chunksize)
    chunksize[1] = 5
    masked_images = masked_images.rechunk(chunksize)
    
    dataset = tiled_client_sandbox.new(
        "sparse",
        COOStructure(
            shape=masked_images.shape,
            chunks=masked_images.chunks,
        ),
    )
    dataset_id = dataset.item['id']
    
    # Run sparsification and write the data to tiled in parallel.
    sparse_images = masked_images.map_blocks(write_sparse_chunk, dataset_id=dataset_id, dataset=dataset).compute()

    return dataset_id


# Make the Prefect Flow.
# A separate command is needed to register it with the Prefect server.
with Flow("sparsify") as flow:
    logger = prefect.context.get("logger")
    logger.info(f"tiled: {tiled.__version__}")
    logger.info(f"databroker: {databroker.__version__}")
    logger.info(f"sparse: {sparse.__version__}")
    logger.info(f"profiles: {tiled.profiles.list_profiles()['nsls2']}")
    ref = Parameter("ref")
    #processed_uid = sparsify(ref)
    #logger.info(f"Processed_uid: {processed_uid.result}")
