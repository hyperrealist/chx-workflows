import numpy as np
import prefect
import sparse
import tiled
import time

from masks import get_mask
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
    detector: string

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
    metadata['detectors'] = [detector]
    # Check if the method below applies to runs in general and not just for run2
    metadata.update(run["primary"].descriptors[0]["configuration"][detector]["data"])

    # Get filename prefix.
    # We think we can still sparsify a run if there are no resources,
    # so we don't raise an exception if no resource is found.
    for name, document in run.documents():
        if name == "resource":
            metadata["filename"] = Path(
                document.get("root", "/"), document["resource_path"]
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
    for key in {"pixel_mask", "binary_mask"}:
        file_metadata[key] = file_metadata[key].tolist()

    metadata.update(file_metadata)

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
    return data


# @task
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
    metadata = get_metadata(run)

    # Load the images.
    dask_images = run["primary"]["data"]["eiger4m_single_image"].read()

    # Rotate the images if he detector is eiger500k_single_image.
    if "eiger500K_single_image" == metadata["detector"]:
        dask_images = np.rotate(dask_images, axis=(3, 2))

    # Get the masks.
    pixel_mask = get_mask("pixel_mask")
    chip_mask = get_mask("eiger4m_chip_mask")
    bad_pixel_mask = get_mask("bad_pixels_4m")
    custom_mask = get_mask("jul11_2022_4m_saxs")
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
    dataset_id = dataset.item["id"]

    # Run sparsification and write the data to tiled in parallel.
    sparse_images = masked_images.map_blocks(
        write_sparse_chunk, dataset_id=dataset_id, dataset=dataset
    ).compute()

    return dataset_id


# Make the Prefect Flow.
# A separate command is needed to register it with the Prefect server.
with Flow("sparsify") as flow:
    logger = prefect.context.get("logger")
    logger.info(f"tiled: {tiled.__version__}")
    logger.info(f"sparse: {sparse.__version__}")
    logger.info(f"profiles: {tiled.profiles.list_profiles()['nsls2']}")
    ref = Parameter("ref")
    # processed_uid = sparsify(ref)
    # logger.info(f"Processed_uid: {processed_uid.result}")
