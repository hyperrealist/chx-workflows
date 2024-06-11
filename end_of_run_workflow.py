from prefect import task, flow, get_run_logger
from data_validation import data_validation
from processing import processing_flow


@task
def log_completion():
    logger = get_run_logger()
    logger.info("Complete")


@flow
def end_of_run_workflow(stop_doc):
    uid = stop_doc["run_start"]
    # return_state = True delays raising exceptions until the end of the validation
    # data_validation(uid, return_state=True)
    processing_flow(uid)
    log_completion()
