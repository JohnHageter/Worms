from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Module.Tracking.Configuration import TrackingParameters
from Module.protocol.runner import TrackingRunner, AppConfig


CONFIG = AppConfig(
    video_dir=Path("D:/Tracking_Data/Sachi/T3/week_1"),
    out_dir=Path("D:/Tracking_Data/Sachi/T3/weel_1_out_v7"),
    roi_path=Path("AgarWell.npy"),
    frame_step=5,
    capture_fps=2,
    save_video=True,
    preview=False,
    foreground_thresh_val=8,
    debug=True,
)

PARAMS = TrackingParameters(
    max_detectable_blobs=5,
    min_blob_area=2,
    max_missed_frames=10,
    max_match_distance=150.0,
    tentative_match_distance=150.0,
)

TrackingRunner(CONFIG, PARAMS).run()
