from tiled.client import from_profile 
from tiled.queries import Key

tiled_client = from_profile("staging", "dask", username=None)["chx"]
tiled_client_sandbox = tiled_client["sandbox"]
results = tiled_client_sandbox.search(Key("spec") == "mask")
vals = results.values()
vals.first()
