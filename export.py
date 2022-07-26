import json
import re
import sys
from pathlib import Path

import httpx
import numpy
import prefect
from prefect import Flow, Parameter, task
from tiled.client import from_profile

EXPORT_PATH = Path("/nsls2/data/dssi/scratch/prefect-outputs/chx/")

tiled_client = from_profile("nsls2", username=None)["chx"]
tiled_client_raw = tiled_client["raw"]
# tiled_client_processed = tiled_client["sandbox"]


def lookup_directory(start_doc, tla="chx"):

    """
    Return the path for the proposal directory.

    PASS gives us a *list* of cycles, and we have created a proposal directory under each cycle.
    """

    DATA_SESSION_PATTERN = re.compile("[pass]*-([0-9]+)")
    client = httpx.Client(base_url="https://api-staging.nsls2.bnl.gov")
    data_session = start_doc[
        "data_session"
    ]  # works on old-style Header or new-style BlueskyRun

    try:
        digits = int(DATA_SESSION_PATTERN.match(data_session).group(1))
    except AttributeError:
        raise AttributeError(f"incorrect data_session: {data_session}")

    response = client.get(f"/proposal/{digits}/directories")
    response.raise_for_status()

    paths = [path_info["path"] for path_info in response.json()]

    # Filter out paths from other beamlines.
    paths = [path for path in paths if tla == path.lower().split("/")[3]]

    # Filter out paths from other cycles and paths for commisioning.
    paths = [
        path
        for path in paths
        if path.lower().split("/")[5] == "commissioning"
        # or path.lower().split("/")[5] == start_doc["cycle"]
    ]

    # There should be only one path remaining after these filters.
    # Convert it to a pathlib.Path.
    return Path(paths[0])


@task
def sparsify(ref):

    """
    Performs sparsification following Mark Sutton's implementation at CHX.

    Parameters
    ----------
    ref: string
        This is the reference to the BlueskyRun to be exported. It can be
        a partial uid, a full uid, a scan_id, or an index (e.g. -1).

    """



# Make the Prefect Flow.
# A separate command is needed to register it with the Prefect server.
with Flow("export") as flow:
    # raw_ref = Parameter("ref")
    # processed_refs = write_dark_subtraction(raw_ref)
