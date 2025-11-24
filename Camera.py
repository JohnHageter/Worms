import sys
import time
import numpy as np
from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl as peak_ipl
import cv2

class Camera:
    def __init__(self, index=0):
        self._index = index
        self._m_device = None
        self._remote = None
        self._datastream = None
        self._is_opened = False
        self._continuous = False
        self._delay_s = 0  # seconds between frames

        self.width = 0
        self.height = 0

    def _open(self):
        if self._is_opened:
            return
        peak.Library.Initialize()
        dev_mgr = peak.DeviceManager.Instance()
        dev_mgr.Update()
        if dev_mgr.Devices().empty():
            print("No camera found")
            sys.exit(-1)
        self._m_device = dev_mgr.Devices()[self._index].OpenDevice(peak.DeviceAccessType_Control)
        self._remote = self._m_device.RemoteDevice().NodeMaps()[self._index]
        self.width = self._remote.FindNode("Width").Value()
        self.height = self._remote.FindNode("Height").Value()
        self._is_opened = True

    def set_delay(self, delay_seconds):
        self._delay_s = delay_seconds
        print(f"Frame delay set to {self._delay_s} seconds")

    def set_attr(self, parameter, value):
        node_map = {
            "exposure": "ExposureTime",
            "gain": "Gain",
            "frame_rate": "AcquisitionFrameRate"
        }
        if parameter not in node_map:
            print(f"Unknown parameter: {parameter}")
            return
        self._open()
        try:
            node = self._remote.FindNode(node_map[parameter])
            min_val, max_val = node.Minimum(), node.Maximum()
            if min_val <= value <= max_val:
                node.SetValue(value)
            else:
                print(f"{parameter} out of bounds [{min_val}-{max_val}]")
        except Exception as e:
            print("EXCEPTION:", e)

    def set_watch_window(self, xoff, width, yoff, height):
        self._open()
        try:
            xoff = max(0, min(xoff, self._remote.FindNode("Width").Maximum()-1))
            yoff = max(0, min(yoff, self._remote.FindNode("Height").Maximum()-1))
            width = min(width, self._remote.FindNode("Width").Maximum()-xoff)
            height = min(height, self._remote.FindNode("Height").Maximum()-yoff)
            self._remote.FindNode("OffsetX").SetValue(xoff)
            self._remote.FindNode("OffsetY").SetValue(yoff)
            self._remote.FindNode("Width").SetValue(width)
            self._remote.FindNode("Height").SetValue(height)
            self.width = width
            self.height = height
            print(f"ROI set to {width}x{height} at ({xoff},{yoff})")
        except Exception as e:
            print("EXCEPTION:", e)

    def _prepare_datastream(self):
        self._datastream = self._m_device.DataStreams()[self._index].OpenDataStream()
        return self._datastream is not None

    def _allocate_buffers(self):
        try:
            payload = self._remote.FindNode("PayloadSize").Value()
            num_buffers = self._datastream.NumBuffersAnnouncedMinRequired()
            for _ in range(num_buffers):
                buf = self._datastream.AllocAndAnnounceBuffer(payload)
                self._datastream.QueueBuffer(buf)
            return True
        except Exception as e:
            print("EXCEPTION in allocation:", e)
            return False

    def _start_acquisition(self):
        try:
            self._datastream.StartAcquisition(peak.AcquisitionStopMode_Default,
                                              peak.DataStream.INFINITE_NUMBER)
            self._remote.FindNode("TLParamsLocked").SetValue(1)
            self._remote.FindNode("AcquisitionStart").Execute()
            self._continuous = True
            print("Continuous acquisition started.")
            return True
        except Exception as e:
            print("EXCEPTION in start:", e)
            return False

    def grab_next_frame(self, timeout_ms=5000, to_bgr=True):
        self._open()

        if self._datastream is None:
            if not self._prepare_datastream():
                raise RuntimeError("Failed to prepare datastream")
            if not self._allocate_buffers():
                raise RuntimeError("Failed to allocate buffers")

        if not self._continuous:
            if not self._start_acquisition():
                raise RuntimeError("Failed to start acquisition")

        try:
            buffer = self._datastream.WaitForFinishedBuffer(timeout_ms)
            img = peak_ipl.Image.CreateFromSizeAndBuffer(
                buffer.PixelFormat(),
                buffer.BasePtr(),
                buffer.Size(),
                buffer.Width(),
                buffer.Height()
            ).get_numpy_2D()
            img = np.ascontiguousarray(img.copy(), dtype=np.uint8)
            self._datastream.QueueBuffer(buffer)

            if self._delay_s >= 0.1:
                self._remote.FindNode("AcquisitionStop").Execute()
                self._remote.FindNode("TLParamsLocked").SetValue(0)
                self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)

            if to_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if self._delay_s >= 0.1:
                time.sleep(self._delay_s)

            return img, True

        except Exception as e:
            print("EXCEPTION grabbing image:", e)
            return None, False


    def close(self):
        try:
            if self._continuous:
                self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
            peak.Library.Close()
            print("Camera closed.")
        except Exception as e:
            print("EXCEPTION closing camera:", e)
