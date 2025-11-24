import sys
import cv2
import numpy as np
from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl as peak_ipl

class Camera:
    def __init__(self, index=0):
        self._index = index
        self._m_device = None
        self._remote = None
        self._datastream = None
        self._is_opened = False

    # ------------------------
    # Internal initialization
    # ------------------------
    def _open(self):
        if self._is_opened:
            return

        try:
            peak.Library.Initialize()
            dev_mgr = peak.DeviceManager.Instance()
            dev_mgr.Update()

            if dev_mgr.Devices().empty():
                print("No camera found.")
                sys.exit(-1)

            self._m_device = dev_mgr.Devices()[self._index].OpenDevice(
                peak.DeviceAccessType_Control)
            self._remote = self._m_device.RemoteDevice().NodeMaps()[self._index]
            self._is_opened = True
            print("Camera initialized.")
        except Exception as e:
            print("EXCEPTION during _open:", e)
            sys.exit(-2)

    # ------------------------
    # Set attributes: exposure, gain, frame_rate
    # ------------------------
    def set_attr(self, parameter, value):
        self._open()
        node_map = {
            "exposure": "ExposureTime",
            "gain": "Gain",
            "frame_rate": "AcquisitionFrameRate"
        }

        if parameter not in node_map:
            print(f"Unknown parameter: {parameter}")
            return

        try:
            node_name = node_map[parameter]
            node = self._remote.FindNode(node_name)
            min_val, max_val = node.Minimum(), node.Maximum()
            if min_val <= value <= max_val:
                node.SetValue(value)
                print(f"{parameter} set to {value}")
            else:
                raise ValueError(f"{parameter} out of bounds ({min_val}-{max_val})")
        except Exception as e:
            print("EXCEPTION in set_attr:", e)

    # ------------------------
    # Load/save camera settings
    # ------------------------
    def load_settings(self, filename):
        self._open()
        try:
            self._remote.LoadFromFile(filename)
            print(f"Settings loaded from {filename}")
        except Exception as e:
            print("EXCEPTION in load_settings:", e)

    def save_settings(self, filename):
        self._open()
        try:
            self._remote.StoreToFile(filename)
            print(f"Settings saved to {filename}")
        except Exception as e:
            print("EXCEPTION in save_settings:", e)

    # ------------------------
    # ROI / Watch window
    # ------------------------
    def set_watch_window(self, xoff, width, yoff, height):
        self._open()
        try:
            ox = self._remote.FindNode("OffsetX")
            oy = self._remote.FindNode("OffsetY")
            w_node = self._remote.FindNode("Width")
            h_node = self._remote.FindNode("Height")

            # Increment validation
            for val, inc, name in [(xoff, ox.Increment(), "OffsetX"),
                                   (yoff, oy.Increment(), "OffsetY"),
                                   (width, w_node.Increment(), "Width"),
                                   (height, h_node.Increment(), "Height")]:
                if val % inc != 0:
                    raise ValueError(f"{name} value {val} not divisible by increment {inc}")

            # Bounds validation
            if not (w_node.Minimum() <= width <= w_node.Maximum()):
                raise ValueError(f"Width {width} out of bounds ({w_node.Minimum()}-{w_node.Maximum()})")
            if not (h_node.Minimum() <= height <= h_node.Maximum()):
                raise ValueError(f"Height {height} out of bounds ({h_node.Minimum()}-{h_node.Maximum()})")

            # Apply
            ox.SetValue(xoff)
            oy.SetValue(yoff)
            w_node.SetValue(width)
            h_node.SetValue(height)

            print(f"ROI set to {width}x{height} at ({xoff},{yoff})")
        except Exception as e:
            print("EXCEPTION in set_watch_window:", e)

    # ------------------------
    # Acquisition helpers
    # ------------------------
    def _prepare_datastream(self):
        try:
            streams = self._m_device.DataStreams()
            if streams.empty():
                raise RuntimeError("No datastreams available")

            self._datastream = streams[self._index].OpenDataStream()
            return True
        except Exception as e:
            print("EXCEPTION in _prepare_datastream:", e)
            return False

    def _allocate_buffers(self):
        try:
            ds = self._datastream
            ds.Flush(peak.DataStreamFlushMode_DiscardAll)
            for buf in ds.AnnouncedBuffers():
                ds.RevokeBuffer(buf)

            payload = self._remote.FindNode("PayloadSize").Value()
            num_buffers = ds.NumBuffersAnnouncedMinRequired()

            for _ in range(num_buffers):
                buf = ds.AllocAndAnnounceBuffer(payload)
                ds.QueueBuffer(buf)

            return True
        except Exception as e:
            print("EXCEPTION in _allocate_buffers:", e)
            return False

    def _start_acquisition(self):
        try:
            self._datastream.StartAcquisition(peak.AcquisitionStopMode_Default,
                                              peak.DataStream.INFINITE_NUMBER)
            self._remote.FindNode("TLParamsLocked").SetValue(1)
            self._remote.FindNode("AcquisitionStart").Execute()
            return True
        except Exception as e:
            print("EXCEPTION in _start_acquisition:", e)
            return False

    # ------------------------
    # Grab a single image as numpy array
    # ------------------------
    def grab_image(self, timeout_ms=5000):
        self._open()
        if not self._prepare_datastream():
            raise RuntimeError("Failed to prepare datastream")
        if not self._allocate_buffers():
            raise RuntimeError("Failed to allocate buffers")
        if not self._start_acquisition():
            raise RuntimeError("Failed to start acquisition")

        try:
            buffer = self._datastream.WaitForFinishedBuffer(timeout_ms)

            # Create IDS IPL image
            ipl_image = peak_ipl.Image.CreateFromSizeAndBuffer(
                buffer.PixelFormat(),
                buffer.BasePtr(),
                buffer.Size(),
                buffer.Width(),
                buffer.Height()
            ).ConvertTo(peak_ipl.PixelFormatName_Mono8,
                        peak_ipl.ConversionMode_Fast).get_numpy_2D()

            # Release buffer back to queue
            self._datastream.QueueBuffer(buffer)

            # Stop acquisition
            self._remote.FindNode("AcquisitionStop").Execute()
            self._remote.FindNode("TLParamsLocked").SetValue(0)
            self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)



            return ipl_image, True

        except Exception as e:
            print("EXCEPTION grabbing image:", e)
            return None, False

    # ------------------------
    # Close camera
    # ------------------------
    def close(self):
        if self._is_opened:
            peak.Library.Close()
            self._is_opened = False
            print("Camera closed.")
