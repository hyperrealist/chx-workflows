import sys
sys.path.insert(0, "/nsls2/data/chx/shared/workflows")

import numpy as np
import prefect
import sparse
import tiled
import time

from functools import reduce
from masks import MaskClient, combine_masks
from pathlib import Path
from prefect import Flow, Parameter, task
from tiled.client import from_profile
from tiled.structures.sparse import COOStructure



EXPORT_PATH = Path("/nsls2/data/dssi/scratch/prefect-outputs/chx/")

# distributed_client = distributed.Client(n_workers=1, threads_per_worker=1, processes=False)
tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
tiled_client_chx = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]


def get_metadata(run):
    """
    Collect the BlueskyRun metadata.

    Parameters
    ----------
    run: BlueskyRun

    Returns
    -------
    metadata: dict
        The BlueskyRun's metadata
    """
    # TODO: Exception or Warning for collisions with the start metadata.

    metadata = {}
    metadata.update(run.start)
    metadata["suid"] = run.start["uid"].split("-")[0]  # short uid
    metadata.update(run.start.get("plan_args", {}))

    # Get the detector metadata.
    detector = run.start["detectors"][0]
    metadata["detector"] = f"{detector}_image"
    metadata["detectors"] = [detector]
    # Check if the method below applies to runs in general and not just for run2
    metadata.update(run["primary"].descriptors[0]["configuration"][detector]["data"])

    # Get filename prefix.
    # We think we can still sparsify a run if there are no resources,
    # so we don't raise an exception if no resource is found.
    for name, document in run.documents():
        if name == "resource":
            metadata["filename"] = str(
                Path(document.get("root", "/"), document["resource_path"])
            )
            break

    if "primary" in run:
        descriptor = run["primary"].descriptors[0]
        # data_keys is a required key in the descriptor document.
        # detector_name must be in the descriptor.data_keys
        metadata["img_shape"] = descriptor["data_keys"][f"{detector}_image"].get(
            "shape", []
        )[:2][::-1]

    # Fix up some datatypes.
    metadata["number of images"] = int(metadata["number of images"])
    metadata["start_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(run.start["time"])
    )
    metadata["stop_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(run.stop["time"])
    )

    # Get detector metadata
    dataset = run[f"{detector}_image_metadata_patched_in_at_runtime"]["data"].read()
    file_metadata = {key: dataset[key].values[0] for key in list(dataset)}

    # Convert numpy arrays to lists.
    # for key in {"pixel_mask", "binary_mask"}:
    #    file_metadata[key] = file_metadata[key].tolist()

    metadata.update(file_metadata)
    del metadata["pixel_mask"]
    del metadata["binary_mask"]

    return metadata


def write_sparse_chunk(data, dataset_id=None, block_info=None, dataset=None):
    result = sparse.COO(data)

    if block_info:
        if dataset is None:
            tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
            tiled_client_sandbox = tiled_client["sandbox"]
            dataset = tiled_client_sandbox[dataset_id]

        dataset.write_block(
            coords=result.coords,
            data=result.data,
            block=block_info[None]["chunk-location"],
        )

    # Returning `data` instead of `result` gives a nice performance
    # improvment.  This causes dask not to update the resulting
    # dataset, which is not needed because we wrote the result
    # to tiled.
    return data

# TODO: Change "chip_mask" back to "pixel_mask"
def sparsify(
    ref,
    mask_names=["chip_mask"]
):
    """
    Performs sparsification.

    Parameters
    ----------
    ref: string
        This is the reference to the BlueskyRun to be exported. It can be
        a partial uid, a full uid, a scan_id, or an index (e.g. -1).
    mask_names: list
        A list of mask names to be applied.

    Returns
    -------
    dataset_uid: string
        The uid of the resulting dataset.
    """
    logger = prefect.context.get("logger")

    # Get the BlueskyRun from Tiled.
    run = tiled_client_chx[ref]

    # Compose the run metadata.
    metadata = get_metadata(run)
    detector_name = metadata["detector"]

    # Load the images.
    images = run["primary"]["data"][detector_name].read()

    # TODO: Save the detector image in the correct orientation,
    # so we don't have to rotate it.
    # Rotate the images if he detector is eiger500k_single_image.
    if detector_name == "eiger500K_single_image":
        images = np.rot90(images, axes=(3, 2))

    # Get the mask.
    mask_client = MaskClient(tiled_client_sandbox)
    uid_masks = [mask_client.get_mask(detector_name, mask_name)
                 for mask_name in mask_names]
    uids = [uid for uid, mask in uid_masks]
    masks = [mask for uid, mask in uid_masks]
    metadata['masks_names'] = mask_names
    metadata['mask_uids'] = uids
    mask = combine_masks(masks)

    # Flip the images.
    images = np.flip(images, axis=2)

    # Apply the mask.
    image_count = images.shape[1]
    mask = np.broadcast_to(mask, (image_count,) + mask.shape)
    images = images * mask

    # Let dask pick the chunk size.
    # Set the block_size_limit equal to the tiled size limit.
    images = images.rechunk(block_size_limit=75_000_000)

    # Create a new dataset in tiled.
    dataset = tiled_client_sandbox.new(
        "sparse",
        COOStructure(
            shape=images.shape,
            chunks=images.chunks,
        ),
        metadata=metadata,
    )
    dataset_id = dataset.item["id"]

    # Run sparsification and write the data to tiled in parallel.
    _ = images.map_blocks(
        write_sparse_chunk, dataset_id=dataset_id, dataset=dataset
    ).compute()

    logger.info(f"dataset_id: {dataset_id}")
    return dataset_id


# Make the Prefect Flow.
# A separate command is needed to register it with the Prefect server.
with Flow("sparsify") as flow:
    logger = prefect.context.get("logger")
    logger.info(f"tiled: {tiled.__version__}")
    logger.info(f"sparse: {sparse.__version__}")
    logger.info(f"profiles: {tiled.profiles.list_profiles()['nsls2']}")
    ref = Parameter("ref")
    # TODO: Change "chip_mask" back to "pixel_mask"
    mask_names = Parameter("mask_names", default=["chip_mask"])
    task(sparsify)(ref,mask_names=mask_names)
