import numpy as np
import pytest
import tiled

from dictdiffer import diff
from chx_compress.io.multifile.multifile import MultifileBNLCustom
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
    "14ed6885-2b6a-4645-85a2-a09413f618c2",
    "961b07ca-2133-46f9-8337-57053baa011b",
    "eaf4d7df-2585-460a-8d9c-5a53613782e7",
    "e909f4a2-12e3-4521-a7a6-be2b728a826b",
)


@pytest.mark.parametrize("run_uid", test_runs)
def test_get_metadata(run_uid):
    """
    Check that the metadata from get_metadata matches the original
    metadata.
    """
    run = tiled_client_chx[run_uid]
    metadata_original = MultifileBNLCustom(
        f"{DATA_DIRECTORY}/uid_{run_uid}.cmp"
    ).md

    metadata_new = get_metadata(run)

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
    original_data = MultifileBNLCustom(
        f"{DATA_DIRECTORY}/uid_{run_uid}.cmp"
    )

    new_data, _ = sparsify(run_uid)

    for frame_number in range(new_data.shape[1]):
        assert np.array_equal(new_data[0][frame_number].todense(),
                              original_data.rdframe(frame_number).astype('uint16'))

