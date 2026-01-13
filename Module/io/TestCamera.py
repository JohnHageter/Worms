from Module.io.Camera import Camera
from Module.io.CameraStates import CameraState


class TestCamera(Camera):
    def __init__(self, index=0):
        super().__init__()

    def open(self):
        self._require_state(CameraState.CLOSED)
        print("Test camera opened")
        self._state = CameraState.OPEN

    def close(self):
        self._require_state(CameraState.OPEN)
        print("Camera closed.")
        self._state = CameraState.CLOSED
