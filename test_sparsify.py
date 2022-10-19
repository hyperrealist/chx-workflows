import numpy as np
import pytest
import tiled

from dictdiffer import diff
from chx_compress.io.multifile.multifile import MultifileBNLCustom
from pandas import Timestamp
from pathlib import Path
from sparsify import get_metadata
from tiled.client import from_profile
from tiled.queries import Key

MULTIFILE_DIRECTORY = Path("/nsls2/data/chx/legacy/Compressed_Data")
tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
tiled_client_chx = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]

run1 = tiled_client_chx["d85d157f-57d9-4649-9b65-0d3b9f754e01"]
run2 = tiled_client_chx["e909f4a2-12e3-4521-a7a6-be2b728a826b"]
run3 = tiled_client_chx["b79184e1-d053-42e4-b1eb-f8ab0a146220"]

run2_file_export = "/nsls2/data/dssi/scratch/prefect-outputs/chx/compressed_data/uid_e909f4a2-12e3-4521-a7a6-be2b728a826b.cmp"
run2_file_chx = "/nsls2/data/chx/legacy/Compression_test/fluerasu/uid_e909f4a2-12e3-4521-a7a6-be2b728a826b.cmp"


def test_get_metadata_run2():
    """
    Check that the metadata from get_metadata matches the original
    metadata.
    """

    metadata_run2_new = get_metadata(run2)
    acceptable_differences = {"filename", "time", "pixel_mask", "binary_mask", ""}

    diffs = diff(metadata_run2_original, metadata_run2_new)
    conflicts = [item for item in diffs if item[1] not in acceptable_differences]

    assert not conflicts


@pytest.mark.parametrize(
    "run_uid",
    (
        "14ed6885-2b6a-4645-85a2-a09413f618c2",
        "961b07ca-2133-46f9-8337-57053baa011b",
        "eaf4d7df-2585-460a-8d9c-5a53613782e7",
        "e909f4a2-12e3-4521-a7a6-be2b728a826b",
    ),
)
def test_get_metadata(run_uid):
    """
    Check that the metadata from get_metadata matches the original
    metadata.
    """
    run = tiled_client_chx[run_uid]
    metadata_original = MultifileBNLCustom(
        f"{MULTIFILE_DIRECTORY}/uid_{run.start['uid']}.cmp"
    ).md

    metadata_new = get_metadata(run)

    exceptions = {'bytes', 'rows_end', 'nrows', 'rows_begin', 'cols_begin', 'ncols', 'cols_end'}

    assert metadata_original.keys() - metadata_new.keys() <= exceptions
    #assert metadata_original.keys() <= metadata_new.keys()

    for key in metadata_original.keys() - exceptions:
        assert metadata_new[key] == metadata_original[key]
