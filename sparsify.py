import databroker
import event_model
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import prefect
import sparse
import time

from collections import defaultdict
from pathlib import Path
from pandas import Timestamp
from prefect import Flow, Parameter, task
from tiled.client import from_profile
from tiled.structures.sparse import COOStructure

EXPORT_PATH = Path("/nsls2/data/dssi/scratch/prefect-outputs/chx/")
MASK_DIR = Path("/nsls2/data/chx/legacy/analysis/masks")

tiled_client = from_profile("nsls2", username=None)["chx"]
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


def get_run_metadata(run, default_dec="eiger", *argv, **kwargs):
    """
    Get metadata from a uid

    - Adds detector key with detector name

    Parameters:
        uid: the unique data acquisition id
        kwargs: overwrite the meta data, for example
            get_run_metadata( uid = uid, sample = 'test') --> will overwrtie the meta's sample to test
    return:
        meta data of the uid: a dictionay
        with keys:
            detector
            suid: the simple given uid
            uid: full uid
            filename: the full path of the data
            start_time: the data acquisition starting time in a human readable manner
        And all the input metadata
    """

    def get_sid_filenames(run):
        """
        Get a bluesky scan_id, unique_id, filename by giveing uid

        Parameters
        ----------
        run: a BlueskyRun

        Returns
        -------
        scan_id: integer
        unique_id: string, a full string of a uid
        filename: string
        """

        filepaths = []
        resources = {}  # uid: document
        datums = defaultdict(list)  # uid: List(document)
        for name, doc in run.documents():
            if name == "resource":
                resources[doc["uid"]] = doc
            elif name == "datum":
                datums[doc["resource"]].append(doc)
            elif name == "datum_page":
                for datum in event_model.unpack_datum_page(doc):
                    datums[datum["resource"]].append(datum)
        for resource_uid, resource in resources.items():
            file_prefix = Path(resource.get("root", "/"), resource["resource_path"])
            if "eiger" not in resource["spec"].lower():
                continue
            for datum in datums[resource_uid]:
                dm_kw = datum["datum_kwargs"]
                seq_id = dm_kw["seq_id"]
                new_filepaths = glob.glob(f"{file_prefix!s}_{seq_id}*")
                filepaths.extend(new_filepaths)
        return run.start["scan_id"], run.start["uid"], filepaths

    def get_devices(run):
        """
        Return the names of the devices in this run.
        Parameters
        ----------
        Returns
        -------
        devices: set
        """
        descriptors = []
        for stream in run.values():
            descriptors.extend(stream.descriptors)

        devices = set()
        for descriptor in descriptors:
            devices.update(descriptor["object_keys"])

        return devices

    def get_detector(run):
        """Get the first detector image name with "eiger" in it."""
        keys = get_detectors(run)
        for k in keys:
            if "eiger" in k:
                return k

    def get_detectors(run):
        """Get all the detector image strings for a run."""
        if "primary" in list(run):
            descriptor = run["primary"].descriptors[0]
            keys = [k for k, v in descriptor["data_keys"].items() if "external" in v]
            return sorted(set(keys))
        else:
            return []

    def get_device_config(run, device_name):

        result = defaultdict(list)

        descriptors = []
        for stream in run.values():
            descriptors.extend(stream.descriptors)

        for descriptor in sorted(descriptors, key=lambda d: d["time"]):
            config = descriptor["configuration"].get(device_name)
            if config:
                result[descriptor.get("name")].append(config["data"])
        return dict(result)["primary"][0]

    if "verbose" in kwargs.keys():  # added: option to suppress output
        verbose = kwargs["verbose"]
    else:
        verbose = True

    md = {}

    md["suid"] = run.start["uid"].split("-")[0]  # short uid
    try:
        md["filename"] = get_sid_filenames(run)[2][0]
    except Exception:
        md["filename"] = "N.A."

    devices = sorted(list(get_devices(run)))
    if len(devices) > 1:
        if verbose:  # added: mute output
            print(
                "More than one device. This would have unintented consequences.Currently, only the device contains 'default_dec=%s'."
                % default_dec
            )

    device_name = devices[0]
    for dec_ in devices:
        if default_dec in dec_:
            device_name = dec_

    # print(dec)
    # detector_names = sorted( header.start['detectors'] )
    detector_names = sorted(get_detectors(run))
    # if len(detector_names) > 1:
    #    raise ValueError("More than one det. This would have unintented consequences.")
    detector_name = detector_names[0]
    # md['detector'] = detector_name
    md["detector"] = get_detector(run)
    # print( md['detector'] )

    new_dict = get_device_config(run, device_name)
    for key, val in new_dict.items():
        newkey = key.replace(detector_name + "_", "")
        md[newkey] = val

    # for k,v in ev['descriptor']['configuration'][dec]['data'].items():
    #     md[ k[len(dec)+1:] ]= v

    try:
        md.update(run.start["plan_args"].items())
        md.pop("plan_args")
    except Exception:
        pass
    md.update(run.start.items())

    # print(header.start.time)
    md["start_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(run.start["time"])
    )
    md["stop_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(run.stop["time"])
    )

    try:  # added: try to handle runs that don't contain image data
        if "primary" in run.keys():
            descriptor = run["primary"].descriptors[0]
            md["img_shape"] = descriptor["data_keys"][md["detector"]]["shape"][:2][::-1]
    except Exception:
        if verbose:
            print("couldn't find image shape...skip!")
        else:
            pass

    md.update(kwargs)

    # Why don't the timestamps match?
    # md['time'] = Timestamp(md['time'], unit='s')

    md["number of images"] = int(md["number of images"])
    return md


@task
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
    if metadata["detector"] == "eiger500K_single_image":
        dask_images = np.rotate(dask_images, axis=(3, 2))

    # Get the masks.
    pixel_mask = get_pixel_mask(metadata)
    chip_mask = get_chip_mask(metadata)
    bad_pixel_mask = get_bad_pixel_mask(metadata)
    custom_mask = get_custom_mask()
    final_mask = pixel_mask & chip_mask & bad_pixel_mask & custom_mask

    # Make the final mask the same shape as the images by extending the mask into a 3d array.
    num_images = dask_images.shape[1]
    mask3d = np.broadcast_to(final_mask, (num_images,) + final_mask.shape)

    # Flip the images to match mask orientation (TODO: probably better to flip the mask for faster computing).
    flipped_images = np.flip(dask_images, axis=2)

    # Apply the mask to flipped images.
    masked_images = flipped_images * mask3d

    # Run sparsification
    sparse_images = masked_images.map_blocks(sparse.COO).compute()

    # Write sparse array to Tiled.
    arr_coo_writer = tiled_client_sandbox.new(
        "sparse",
        COOStructure(
            shape=sparse_images.shape,
            chunks=(
                (1,),
                (1,) * num_images,
                (sparse_images.shape[2],),
                (sparse_images.shape[3],),
            ),
        ),
    )

    for block_i, block_start in enumerate(range(0, num_images)):
        arr_coo_writer.write_block(
            coords=sparse_images[:, block_start : block_start + 1].coords,
            data=sparse_images[:, block_start : block_start + 1].data,
            block=(0, block_i, 0, 0),
        )

    processed_uid = arr_coo_writer._item["id"]
    return processed_uid


# Make the Prefect Flow.
# A separate command is needed to register it with the Prefect server.
with Flow("sparsify") as flow:
    ref = Parameter("ref")
    processed_refs = sparsify(ref)
