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
    "14ed6885-2b6a-4645-85a2-a09413f618c2", # Just fixed the rootmap.
    "961b07ca-2133-46f9-8337-57053baa011b", # Very dense. Working.
    "eaf4d7df-2585-460a-8d9c-5a53613782e7", # Need the correct masks.
    "e909f4a2-12e3-4521-a7a6-be2b728a826b", # Working.
    "1bdfe080-250e-4874-b4cc-0b5acbdc99a3",
    "2e902219-5910-434f-9c89-75d7ea46db44",
    "4b31b18c-6207-491e-bca7-162431281a89",
    "5f1dc5a6-8d9f-479a-b2b5-5e2fabec594e",
    "6dff7417-6d41-4e71-ad3e-6982e838578f",
    "7d95f576-dfec-4bc7-905d-6b801420ca48",
    "81bcc380-d5cb-49f6-821d-4ca5ba0407e2",
    "8a8b73b8-5b8e-4dcd-81cd-c211520dccfb",
    "8af27b39-dd4c-4630-bdff-21af77b22227",
    "8ffddd5b-31f7-433f-babb-a2c995fd2587",
    "f9d3b730-1915-455e-985c-bb169fb74876",
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

    # TODO: Update chx_patches in tiled_site_config to get this metadata.
    exceptions = {
        "bytes",
        "byte_count",
        "rows_end",
        "nrows",
        "rows_begin",
        "cols_begin",
        "ncols",
        "cols_end",
    }

    assert metadata_original.keys() - metadata_new.keys() <= exceptions

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
    Make sure that the new multifile matches the original multifile.
    """
    original_file = f"{DATA_DIRECTORY}/uid_{run_uid}.cmp"
    processed_uid = sparsify(run_uid)
    new_file = f"/tmp/{processed_uid}.cmp"
    processed_data = tiled_client_sandbox[processed_uid]
    processed_data.export(new_file, format='application/x-eiger-multifile')
    assert filecmp.cmp(original_file, new_file)
