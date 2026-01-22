import numpy as np
import cv2

from ids_peak import ids_peak as peak  # type: ignore
from ids_peak_ipl import ids_peak_ipl as peak_ipl  # type: ignore

from Module.io.Camera import Camera as Camera


class IDSCamera(Camera):
    def __init__(self, index: int = 0):
        super().__init__()
        self._index = index

        self._m_device: peak.Device
        self._remote_device: peak.RemoteDevice
        self._remote_map: peak.NodeMap
        self._datastream: peak.DataStream

        self.width: float = 0
        self.height: float = 0
        self._continuous: bool = False
        self._delay_s: float = 0.0

    def _do_open(self) -> None:
        peak.Library.Initialize()
        dev_mgr = peak.DeviceManager.Instance()
        dev_mgr.Update()

        if dev_mgr.Devices().empty():
            raise RuntimeError("No IDS Peak camera found")

        self._m_device = dev_mgr.Devices()[self._index].OpenDevice(
            peak.DeviceAccessType_Control
        )
        self._remote_device = self._m_device.RemoteDevice()
        self._remote_map = self._remote_device.NodeMaps()[self._index]

        self.width = self._remote_map.FindNode("Width").Value() 
        self.height = self._remote_map.FindNode("Height").Value()

    def _do_close(self) -> None:
        if self._continuous:
            self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
            self._continuous = False
        peak.Library.Close()

    def _do_set_parameter(self, key: str, value: float) -> None:
        node_map = {
            "exposure": "ExposureTime",
            "gain": "Gain",
            "frame_rate": "AcquisitionFrameRate",
        }
        if key not in node_map:
            raise ValueError(f"Unknown parameter: {key}")

        node = self._remote_map.FindNode(node_map[key])
        min_val, max_val = node.Minimum(), node.Maximum()
        if not (min_val <= value <= max_val):
            raise ValueError(f"{key} out of bounds [{min_val}-{max_val}]")
        node.SetValue(value)

    def _do_set_framerate(self, fps: float) -> None:
        self._do_set_parameter("frame_rate", fps)

    def _do_set_watch_window(
        self, x_start: float, width: float, y_start: float, height: float
    ) -> None:
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

        self.width = width
        self.height = height

    def _prepare_datastream(self) -> None:
        self._datastream = self._m_device.DataStreams()[self._index].OpenDataStream()

    def _allocate_buffers(self) -> None:
        payload = self._remote_map.FindNode("PayloadSize").Value()
        num_buffers = self._datastream.NumBuffersAnnouncedMinRequired()
        for _ in range(num_buffers):
            buf = self._datastream.AllocAndAnnounceBuffer(payload)
            self._datastream.QueueBuffer(buf)

    def _do_start_stream(self) -> None:
        if self._datastream is None:
            self._prepare_datastream()
            self._allocate_buffers()

        self._datastream.StartAcquisition(
            peak.AcquisitionStopMode_Default, peak.DataStream.INFINITE_NUMBER
        )
        self._remote_map.FindNode("TLParamsLocked").SetValue(1)
        self._remote_map.FindNode("AcquisitionStart").Execute()
        self._continuous = True

    def _do_stop_stream(self) -> None:
        if self._continuous:
            self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
            self._remote_map.FindNode("TLParamsLocked").SetValue(0)
            self._continuous = False

    def _do_grab_frame(self) -> np.ndarray:
        if self._datastream is None:
            self._prepare_datastream()
            self._allocate_buffers()
            self._do_start_stream()

        buffer = self._datastream.WaitForFinishedBuffer(peak.DataStream.INFINITE_NUMBER)
        img = peak_ipl.Image.CreateFromSizeAndBuffer(
            buffer.PixelFormat(),
            buffer.BasePtr(),
            buffer.Size(),
            buffer.Width(),
            buffer.Height(),
        ).get_numpy_2D()
        self._datastream.QueueBuffer(buffer)

        img_array = np.ascontiguousarray(img.copy(), dtype=np.uint8)
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    def _do_grab_last_frame(self) -> np.ndarray:
        # Same as _do_grab_frame, but without starting acquisition if not streaming
        buffer = self._datastream.WaitForFinishedBuffer(5000)
        img = peak_ipl.Image.CreateFromSizeAndBuffer(
            buffer.PixelFormat(),
            buffer.BasePtr(),
            buffer.Size(),
            buffer.Width(),
            buffer.Height(),
        ).get_numpy_2D()
        self._datastream.QueueBuffer(buffer)

        img_array = np.ascontiguousarray(img.copy(), dtype=np.uint8)
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
