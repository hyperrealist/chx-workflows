import filecmp
import numpy as np
import pytest
import tiled

from dictdiffer import diff
from chx_compress.io.multifile.multifile import multifile_reader
from pandas import Timestamp
from pathlib import Path
from sparsify import get_metadata, sparsify
from tiled.client import from_profile
from tiled.queries import Key

DATA_DIRECTORY = Path("/nsls2/data/chx/legacy/Compressed_Data")
tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
tiled_client_chx = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]

run1 = tiled_client_chx["d85d157f-57d9-4649-9b65-0d3b9f754e01"]
run2 = tiled_client_chx["e909f4a2-12e3-4521-a7a6-be2b728a826b"]
run3 = tiled_client_chx["b79184e1-d053-42e4-b1eb-f8ab0a146220"]

test_runs = (
    "14ed6885-2b6a-4645-85a2-a09413f618c2", # Needs rootmap
    "961b07ca-2133-46f9-8337-57053baa011b", # Very dense. Working.
    "eaf4d7df-2585-460a-8d9c-5a53613782e7", # Need the correct masks.
    "e909f4a2-12e3-4521-a7a6-be2b728a826b", # Working.
)


def read_frame(multifile, image_index):
    image_array = np.zeros((multifile.header_info['ncols'],
                            multifile.header_info['nrows']))
    np.put(image_array, *multifile[image_index])
    return image_array.astype('uint16')


@pytest.mark.parametrize("run_uid", test_runs)
def test_get_metadata(run_uid):
    """
    Check that the metadata from get_metadata matches the original
    metadata.
    """
 
    run = tiled_client_chx[run_uid]
    metadata_new = get_metadata(run)
    
    original_data = multifile_reader(
        f"{DATA_DIRECTORY}/uid_{run_uid}.cmp"
    ).header_info

    # TODO: figure out if these exceptions are needed.
    # And figure out how these values get into the Multifile.
    exceptions = {
        "bytes",
        "rows_end",
        "nrows",
        "rows_begin",
        "cols_begin",
        "ncols",
        "cols_end",
    }

    assert metadata_original.keys() - metadata_new.keys() <= exceptions
    # assert metadata_original.keys() <= metadata_new.keys()

    for key in metadata_original.keys() - exceptions:
        assert metadata_new[key] == metadata_original[key]


@pytest.mark.parametrize("run_uid", test_runs)
def test_sparsify(run_uid):
    """
    Make sure that the processed data from sparisfy  
    matches the original proccesed data.
    """
    original_data = multifile_reader(
        f"{DATA_DIRECTORY}/uid_{run_uid}.cmp"
    )

    processed_uid = sparsify(run_uid)
    new_data = tiled_client_sandbox[processed_uid]

    for frame_number in range(new_data.shape[1]):
        assert np.array_equal(new_data[0][frame_number].todense(),
                              read_frame(original_data, frame_number))


@pytest.mark.parametrize("run_uid", test_runs)
def test_multifile(run_uid):
    """
    Make sure that the processed data from sparisfy  
    matches the original proccesed data.
    """
    original_file = f"{DATA_DIRECTORY}/uid_{run_uid}.cmp"
    new_file = f"/tmp/{processed_uid}.cmp"
    processed_uid = sparsify(run_uid)
    processed_data = tiled_client_sandbox[processed_uid]
    processed_data.export(new_file, format='application/x-eiger-multifile')
    assert filecmp.cmp(original_file, new_file)
