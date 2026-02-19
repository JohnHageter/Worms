import numpy as np
import cv2
from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl as peak_ipl
from Module.io.Camera import Camera, CameraError
from typing import Optional, Tuple


class IDSCamera(Camera):
    def __init__(self, index=0):
        super().__init__()
        peak.Library.Initialize()
        self._index = index
        self.device_manager = peak.DeviceManager.Instance()
        self._m_device = None
        self._remote_device = None
        self._remote_map = None
        self._datastream = None
        self._continuous_acquisition = False
        self._single_acquisition = True
        self._latest_frame: Optional[np.ndarray] = None

    def _open(self) -> None:
        self.device_manager.Update()
        if self.device_manager.Devices().empty():
            raise CameraError("No IDS Peak camera found")

        self._m_device = self.device_manager.Devices()[self._index].OpenDevice(
            peak.DeviceAccessType_Control
        )

        self._remote_device = self._m_device.RemoteDevice()
        self._remote_map = self._remote_device.NodeMaps()[self._index]
        self.width = self._remote_map.FindNode("Width").Value()
        self.height = self._remote_map.FindNode("Height").Value()
        self.watch_window = 0, self.width, 0, self.height
        self.cam = self._m_device

        self._datasteam = None
        self._latest_frame = None
        self._continuous = False

    def _close(self) -> None:
        del self._m_device
        peak.Library.Close()

    def _set(self, key, value) -> bool:
        if not self._remote_map:
            return False

        if key == "exposure":
            # IDS sets exposure in nanoseconds
            value = value * 1e6
            min_exposure = self._remote_map.FindNode("ExposureTime").Minimum()
            max_exposure = self._remote_map.FindNode("ExposureTime").Maximum()
            if value < max_exposure and value > min_exposure:
                self._remote_map.FindNode("ExposureTime").SetValue(value)
                return True
            return False
        elif key == "gain":
            node = self._remote_map.FindNode("Gain")
            min_gain, max_gain = node.Minimum(), node.Maximum()
            if min_gain < value < max_gain:
                node.SetValue(value)
                return True
            return False
        elif key == "fps":
            ret = self._set_framerate(value)
            return ret
        else:
            return False

    def _set_framerate(self, fps) -> bool:
        if fps <= 0:
            raise CameraError("Frame rate must be above 0")

        if not self._remote_map:
            raise CameraError("Camera not open.")

        # We just free run the camera and make sure that we're not exceeding hardware capabilities.
        # limit frame rate by pulling at necessary intervals elsewhere
        try:
            node = self._remote_map.FindNode("AcquisitionFrameRate")
            max_fps = node.Maximum()
            min_fps = node.Minimum()

            fps = max(min(fps, max_fps), min_fps)
            node.SetValue(fps)
            applied = node.Value()
            self._camera_fps = applied

            return abs(applied - fps) < 1e3
        except Exception:
            return False

    def _watch(self, x_start, width, y_start, height):
        if not self._remote_map:
            return

        x_start = max(0, min(x_start, self._remote_map.FindNode("Width").Maximum() - 1))
        y_start = max(
            0, min(y_start, self._remote_map.FindNode("Height").Maximum() - 1)
        )
        width = min(width, self._remote_map.FindNode("Width").Maximum() - x_start)
        height = min(height, self._remote_map.FindNode("Height").Maximum() - y_start)
        self._remote_map.FindNode("OffsetX").SetValue(x_start)
        self._remote_map.FindNode("OffsetY").SetValue(y_start)
        self._remote_map.FindNode("Width").SetValue(width)
        self._remote_map.FindNode("Height").SetValue(height)
        self.watch_window = x_start, width, y_start, height

    def _reset_watch_window(self):
        if not self._remote_map:
            return

        x_start = 0
        y_start = 0
        width = self._remote_map.FindNode("Width").Maximum()
        height = self._remote_map.FindNode("Height").Maximum()
        self._remote_map.FindNode("OffsetX").SetValue(x_start)
        self._remote_map.FindNode("OffsetY").SetValue(y_start)
        self._remote_map.FindNode("Width").SetValue(width)
        self._remote_map.FindNode("Height").SetValue(height)
        self.watch_window = x_start, width, y_start, height

    def _read_frame(self) -> Tuple[bool, np.ndarray | None]:
        if not self._m_device or not self._remote_map:
            return False, None

        try:
            if not self._datastream:
                self._prepare_datastream()

            frame = self._do_grab_frame(timeout_s=1.0)

            if frame is None:
                return False, None

            return True, frame

        except Exception:
            return False, None

    ### IDS Helpers for base class

    def _prepare_datastream(self):
        if not self._m_device or not self._remote_map:
            return
        self._datastream = self._m_device.DataStreams()[self._index].OpenDataStream()
        payload = self._remote_map.FindNode("PayloadSize").Value()
        num_buffers = self._datastream.NumBuffersAnnouncedMinRequired()
        for _ in range(num_buffers):
            buf = self._datastream.AllocAndAnnounceBuffer(payload)
            self._datastream.QueueBuffer(buf)

    def _do_start_stream(self):
        if not self._datastream:
            self._prepare_datastream()

        assert self._datastream is not None
        assert self._remote_map is not None
        self._datastream.StartAcquisition(
            peak.AcquisitionStopMode_Default, peak.DataStream.INFINITE_NUMBER
        )
        self._remote_map.FindNode("TLParamsLocked").SetValue(1)
        self._remote_map.FindNode("AcquisitionStart").Execute()
        self._continuous = True

    def _do_stop_stream(self):
        if self._continuous:
            if not self._datastream or not self._remote_map:
                return
            self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
            self._remote_map.FindNode("TLParamsLocked").SetValue(0)
            self._continuous = False

    def _do_grab_frame(self, timeout_s: float):
        if not self._remote_map or not self._datastream:
            return

        if not self._continuous:
            self._do_start_stream()
        buffer = self._datastream.WaitForFinishedBuffer(int(timeout_s * 1000))
        img = peak_ipl.Image.CreateFromSizeAndBuffer(
            buffer.PixelFormat(),
            buffer.BasePtr(),
            buffer.Size(),
            buffer.Width(),
            buffer.Height(),
        ).get_numpy_2D()
        self._datastream.QueueBuffer(buffer)
        img_array = np.ascontiguousarray(img.copy(), dtype=np.uint8)
        frame = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        self._latest_frame = frame
        return frame
