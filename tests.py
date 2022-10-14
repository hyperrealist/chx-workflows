import numpy as np
import tiled

from dictdiffer import diff
from pandas import Timestamp
from pathlib import Path
from sparsify import get_metadata
from tiled.client import from_profile
from tiled.queries import Key

tiled_client = from_profile("nsls2", "dask", username=None)["chx"]
tiled_client_chx = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]

run1 = tiled_client_chx["d85d157f-57d9-4649-9b65-0d3b9f754e01"]
run2 = tiled_client_chx["e909f4a2-12e3-4521-a7a6-be2b728a826b"]
run3 = tiled_client_chx["b79184e1-d053-42e4-b1eb-f8ab0a146220"]

run2_file_export = "/nsls2/data/dssi/scratch/prefect-outputs/chx/compressed_data/uid_e909f4a2-12e3-4521-a7a6-be2b728a826b.cmp"
run2_file_chx = "/nsls2/data/chx/legacy/Compression_test/fluerasu/uid_e909f4a2-12e3-4521-a7a6-be2b728a826b.cmp"

run2_chx_md = {
    "suid": "e909f4a2",
    "filename": "/nsls2/data/chx/legacy/data/2022/07/18/64d6bc55-2b1e-45eb-b5ba_1216_data_000020.h5",
    "detector": "eiger4m_single_image",
    "eiger4m_single_cam_acquire_period": 0.002,
    "eiger4m_single_cam_acquire_time": 0.001997,
    "eiger4m_single_cam_num_images": 200,
    "eiger4m_single_beam_center_x": 1143.0,
    "eiger4m_single_beam_center_y": 1229.0,
    "eiger4m_single_wavelength": 1.28507668359453,
    "eiger4m_single_det_distance": 16.235550295,
    "eiger4m_single_threshold_energy": 4824.0,
    "eiger4m_single_photon_energy": 9648.0,
    "eiger4m_single_stats1_bgd_width": 1,
    "eiger4m_single_stats1_centroid_threshold": 1.0,
    "eiger4m_single_stats1_compute_centroid": "No",
    "eiger4m_single_stats1_compute_histogram": "No",
    "eiger4m_single_stats1_compute_profiles": "No",
    "eiger4m_single_stats1_compute_statistics": "Yes",
    "eiger4m_single_stats1_hist_max": 255.0,
    "eiger4m_single_stats1_hist_min": 0.0,
    "eiger4m_single_stats1_hist_size": 256,
    "eiger4m_single_stats1_ts_num_points": 2048,
    "eiger4m_single_stats2_bgd_width": 1,
    "eiger4m_single_stats2_centroid_threshold": 1.0,
    "eiger4m_single_stats2_compute_centroid": "No",
    "eiger4m_single_stats2_compute_histogram": "No",
    "eiger4m_single_stats2_compute_profiles": "No",
    "eiger4m_single_stats2_compute_statistics": "Yes",
    "eiger4m_single_stats2_hist_max": 255.0,
    "eiger4m_single_stats2_hist_min": 0.0,
    "eiger4m_single_stats2_hist_size": 256,
    "eiger4m_single_stats2_ts_num_points": 2048,
    "eiger4m_single_stats3_bgd_width": 1,
    "eiger4m_single_stats3_centroid_threshold": 1.0,
    "eiger4m_single_stats3_compute_centroid": "No",
    "eiger4m_single_stats3_compute_histogram": "No",
    "eiger4m_single_stats3_compute_profiles": "No",
    "eiger4m_single_stats3_compute_statistics": "Yes",
    "eiger4m_single_stats3_hist_max": 255.0,
    "eiger4m_single_stats3_hist_min": 0.0,
    "eiger4m_single_stats3_hist_size": 256,
    "eiger4m_single_stats3_ts_num_points": 2048,
    "eiger4m_single_stats4_bgd_width": 1,
    "eiger4m_single_stats4_centroid_threshold": 1.0,
    "eiger4m_single_stats4_compute_centroid": "No",
    "eiger4m_single_stats4_compute_histogram": "No",
    "eiger4m_single_stats4_compute_profiles": "No",
    "eiger4m_single_stats4_compute_statistics": "Yes",
    "eiger4m_single_stats4_hist_max": 255.0,
    "eiger4m_single_stats4_hist_min": 0.0,
    "eiger4m_single_stats4_hist_size": 256,
    "eiger4m_single_stats4_ts_num_points": 2048,
    "eiger4m_single_stats5_bgd_width": 1,
    "eiger4m_single_stats5_centroid_threshold": 1.0,
    "eiger4m_single_stats5_compute_centroid": "No",
    "eiger4m_single_stats5_compute_histogram": "No",
    "eiger4m_single_stats5_compute_profiles": "No",
    "eiger4m_single_stats5_compute_statistics": "Yes",
    "eiger4m_single_stats5_hist_max": 255.0,
    "eiger4m_single_stats5_hist_min": 0.0,
    "eiger4m_single_stats5_hist_size": 256,
    "eiger4m_single_stats5_ts_num_points": 2048,
    "detectors": ["eiger4m_single"],
    "num": 1,
    "uid": "e909f4a2-12e3-4521-a7a6-be2b728a826b",
    "time": Timestamp("2022-08-17 17:51:31.957174528"),
    "owner": "xf11id",
    "sample": "membrane",
    "cycle": "2022_2",
    "scan_id": 101172,
    "user": "rama",
    "beamline_id": "CHX",
    "auto_pipeline": "XPCS_SAXS_2022_2_v1",
    "plan_type": "generator",
    "plan_name": "count",
    "num_points": 1,
    "num_intervals": 0,
    "plan_args": {
        "detectors": [
            "EigerSingleTrigger_AD37_V2(prefix='XF:11IDB-ES{Det:Eig4M}', name='eiger4m_single', read_attrs=['file', 'stats1', 'stats1.total', 'stats2', 'stats2.total', 'stats3', 'stats3.total', 'stats4', 'stats4.total', 'stats5', 'stats5.total'], configuration_attrs=['cam', 'cam.acquire_period', 'cam.acquire_time', 'cam.num_images', 'file', 'beam_center_x', 'beam_center_y', 'wavelength', 'det_distance', 'threshold_energy', 'photon_energy', 'stats1', 'stats1.bgd_width', 'stats1.centroid_threshold', 'stats1.compute_centroid', 'stats1.compute_histogram', 'stats1.compute_profiles', 'stats1.compute_statistics', 'stats1.hist_max', 'stats1.hist_min', 'stats1.hist_size', 'stats1.profile_cursor', 'stats1.profile_size', 'stats1.ts_num_points', 'stats2', 'stats2.bgd_width', 'stats2.centroid_threshold', 'stats2.compute_centroid', 'stats2.compute_histogram', 'stats2.compute_profiles', 'stats2.compute_statistics', 'stats2.hist_max', 'stats2.hist_min', 'stats2.hist_size', 'stats2.profile_cursor', 'stats2.profile_size', 'stats2.ts_num_points', 'stats3', 'stats3.bgd_width', 'stats3.centroid_threshold', 'stats3.compute_centroid', 'stats3.compute_histogram', 'stats3.compute_profiles', 'stats3.compute_statistics', 'stats3.hist_max', 'stats3.hist_min', 'stats3.hist_size', 'stats3.profile_cursor', 'stats3.profile_size', 'stats3.ts_num_points', 'stats4', 'stats4.bgd_width', 'stats4.centroid_threshold', 'stats4.compute_centroid', 'stats4.compute_histogram', 'stats4.compute_profiles', 'stats4.compute_statistics', 'stats4.hist_max', 'stats4.hist_min', 'stats4.hist_size', 'stats4.profile_cursor', 'stats4.profile_size', 'stats4.ts_num_points', 'stats5', 'stats5.bgd_width', 'stats5.centroid_threshold', 'stats5.compute_centroid', 'stats5.compute_histogram', 'stats5.compute_profiles', 'stats5.compute_statistics', 'stats5.hist_max', 'stats5.hist_min', 'stats5.hist_size', 'stats5.profile_cursor', 'stats5.profile_size', 'stats5.ts_num_points'])"
        ],
        "num": 1,
    },
    "hints": {"dimensions": [[["time"], "primary"]]},
    "exposure time": "0.002",
    "acquire period": "0.002",
    "shutter mode": "single",
    "number of images": 200,
    "data path": "/nsls2/data/chx/legacy/data/2022/07/18/",
    "sequence id": "1216",
    "transmission": 1.0,
    "OAV_mode": "none",
    "T_yoke": "16.591",
    "T_sample": "16.591",
    "T_sample_stinger": "0.0",
    "analysis": "q_phi",
    "feedback_x": "on",
    "feedback_y": "on",
    "T_yoke_": "177.005",
    "OAV_mode_": "none",
    "feedback_y_": "on",
    "number of images_": "500",
    "shutter mode_": "single",
    "T_sample_stinger_": "0.0",
    "sequence id_": "1165",
    "T_sample_": "155.667",
    "transmission_": 0.0012954448758557566,
    "analysis_": "iso",
    "feedback_x_": "on",
    "exposure time_": "2.5",
    "data path_": "/nsls2/data/chx/legacy/data/2022/07/15/",
    "acquire period_": "2.5",
    "Measurement": "XPCS test 4m",
    "start_time": "2022-07-18 15:25:36",
    "stop_time": "2022-07-18 15:25:40",
    "img_shape": [2167, 200],
    "y_pixel_size": 7.5e-05,
    "x_pixel_size": 7.5e-05,
    "detector_distance": 16.235550295,
    "incident_wavelength": 1.28507668359453,
    "frame_time": 0.002,
    "beam_center_x": 1143.0,
    "beam_center_y": 1229.0,
    "count_time": 0.001997,
}


def test_run2_get_metadata():
    """
    This test compares metadata of run2 from CHX analysis and
    DAW workflow
    """

    run2_daw_md = get_metadata(run2)
    del run2_daw_md['pixel_mask']
    del run2_daw_md['binary_mask']

    acceptable_differences = ['filename', 'time']

    diffs = diff(run2_chx_md, run2_daw_md)
    conflicts = [item for item in diffs if item[1] not in acceptable_differences]

    assert not conflicts

