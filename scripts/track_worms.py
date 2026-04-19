from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.Tracking.Configuration import TrackingParameters
from Module.protocol.runner import TrackingRunner, AppConfig

import numpy as np


CONFIG = AppConfig(
    video_dir=Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/first"),
    out_dir=Path(
        "D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/out_v2"
    ),  # new folder
    frame_step=2,
    capture_fps=2,
    save_video=True,
    preview=True,
    foreground_thresh_val=10,  # may need tuning (try 15‑25)
    debug=True,
    coords_are_full_frame=False,
)

PARAMS = TrackingParameters(
    max_detectable_blobs=100,
    min_blob_area=8,
    min_skeleton_length=20.0,
    max_missed_frames=60,  # delete after 30 seconds
    max_match_distance=80.0,  # worms move far at 2 fps
    min_speed_for_headtail=5.0,
    headtail_gain_motion=0.6,
    headtail_penalty_motion=0.8,
    headtail_gain_shape=0.15,
    headtail_penalty_shape=0.15,
    headtail_gain_persist=0.1,
    reid_grace_frames=200,  # remember lost tracks for 100 seconds
    reid_max_distance=120.0,
    reid_area_weight=100,
    wall_reid_grace_frames=1000,  # 500 seconds for wall crawlers
    wall_reid_max_distance=200.0,
    birth_suppression_dist=60.0,
    birth_suppression_iou=0.4,
    tentative_match_distance=100.0,
    require_motion=False,  # suppress static ghosts
    debug=True,
)

TrackingRunner(CONFIG, PARAMS).run()
