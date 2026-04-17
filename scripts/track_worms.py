from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.Tracking.Configuration import TrackingParameters
from Module.protocol.runner import TrackingRunner, AppConfig

import numpy as np


CONFIG = AppConfig(
    video_dir=Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/first"),
    out_dir=Path("D:/Tracking_Data/Sachi/T4_4233A_Rig_AgaroseWell/out_v1"),
    frame_step=2,
    capture_fps=2,
    save_video=True,
    preview=False,
    foreground_thresh_val=20,
    debug=True,
)

PARAMS = TrackingParameters(
    max_detectable_blobs=2000,  # limit max number. 20 is good because sometimes single worm gets new ID by accident
    min_blob_area=12,  # minimum pixel area for worms to count
    min_skeleton_length=20.0,  # minimum length in pixels for head and tail to appear
    max_missed_frames=10000000,  # Just never drop tracks. we'll match them later
    max_match_distance=20.0,  # max distance between frames for worms to be considered the same
    min_speed_for_headtail=5.0,  # assigns head and tail if the speed exceeds this value
    headtail_gain_motion=0.6,
    headtail_penalty_motion=0.8,
    headtail_gain_shape=0.15,
    headtail_penalty_shape=0.15,
    headtail_gain_persist=0.1,
    reid_grace_frames=1,  # how many frames if a worm goes missing should we give it before that track dies
    reid_max_distance=20.0,  # max distance to match an ID if it went missing and came back
    reid_area_weight=100,  # matching worms should roughly be the same shape
    wall_reid_grace_frames=1000,  # differnt grace duration if they're on the well wall
    wall_reid_max_distance=100,
    debug=True,
)

TrackingRunner(CONFIG, PARAMS).run()
