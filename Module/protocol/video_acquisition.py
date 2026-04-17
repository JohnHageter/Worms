import time
import os
from pathlib import Path

from Module.protocol.well_processor import Protocol
from Module.io.Camera import Camera
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np

@dataclass(slots=True)
class VideoProtocolConfig:
    output_dir: Path
    
    single_video_duration: float
    num_videos: int
    interval: float
    prefix: str
    
    camera: Camera
    exposure: float
    gain: float
    target_fps: float
    watch_window: Tuple[float,float,float,float] | None


class VideoAcquisition(Protocol):
    def __init__(self, camera: Camera, config: VideoProtocolConfig):
        super().__init__(camera)
        self.config = config
        self.camera = camera

    def _run(self):
        os.makedirs(self.config.output_dir, exist_ok=True)

        camera = self.config.camera
        camera.open()
        camera.set("exposure", self.config.exposure)
        camera.set("gain", self.config.gain)

        if self.config.watch_window:
            x,w,y,h = self.config.watch_window
            camera.watch(x,w,y,h)

        try:
            for i in range(self.config.num_videos):
                start_time = time.time()

                filename = f"{self.config.prefix}_{i:04d}.mp4"
                output_path = self.config.output_dir / filename

                _, frame, _ = camera.read()
                if frame is None:
                    raise RuntimeError("Camera returned no frame.")

                height, width = frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.config.target_fps,
                    (width, height),
                    isColor=(len(frame.shape) == 3),
                )

                frame_interval = 1.0 / self.config.target_fps
                next_frame_time = time.time()

                while (time.time() - start_time) < self.config.single_video_duration:
                    now = time.time()

                    if now >= next_frame_time:
                        _, frame, ts = camera.read()
                        if frame is None:
                            break

                        if frame.dtype != np.uint8:
                            dst = np.empty_like(frame, dtype=np.uint8)
                            frame = cv2.normalize(
                                frame, dst, 0, 255, cv2.NORM_MINMAX
                            ).astype(np.uint8)

                        writer.write(frame)
                        next_frame_time += frame_interval
                    else:
                        time.sleep(0.001)

                writer.release()

                if i < self.config.num_videos - 1:
                    time.sleep(self.config.interval)

        finally:
            camera.close()
