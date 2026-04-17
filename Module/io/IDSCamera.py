from typing import Dict, Optional, Tuple, Type, Union, overload
from typing_extensions import Literal
from typing import cast

import numpy as np
import cv2

from ids_peak import ids_peak as peak
from ids_peak_ipl import ids_peak_ipl as peak_ipl

from Module.io.Camera import (
    Camera,
    CameraError,
    InvalidStateError,
    Parameter,
)


CameraValue = Union[int, float]

# ---------------------------------------------------------------------
# Node registries
# ---------------------------------------------------------------------

_NODE_REGISTRY: Dict[str, Type[peak.Node]] = {
    "AcquisitionFrameRate": peak.FloatNode,
    "AcquisitionMode": peak.EnumerationNode,
    "AcquisitionStart": peak.CommandNode,
    "AcquisitionStop": peak.CommandNode,
    "ExposureAuto": peak.EnumerationNode,
    "ExposureMode": peak.EnumerationNode,
    "ExposureTime": peak.FloatNode,
    "Gain": peak.FloatNode,
    "GainSelector": peak.EnumerationNode,
    "GainAuto": peak.EnumerationNode,
    "OffsetX": peak.IntegerNode,
    "OffsetY": peak.IntegerNode,
    "Width": peak.IntegerNode,
    "Height": peak.IntegerNode,
    "TLParamsLocked": peak.IntegerNode,
    "UserSetSelector": peak.EnumerationNode,
    "UserSetLoad": peak.CommandNode,
}

_NODETYPE_TO_CLASS: Dict[int, Type[peak.Node]] = {
    peak.NodeType_Integer: peak.IntegerNode,
    peak.NodeType_Float: peak.FloatNode,
    peak.NodeType_Enumeration: peak.EnumerationNode,
    peak.NodeType_Command: peak.CommandNode,
    peak.NodeType_Boolean: peak.BooleanNode,
    peak.NodeType_String: peak.StringNode,
}


class IDSCamera(Camera):

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def __init__(self, index: int = 0) -> None:
        super().__init__()

        peak.Library.Initialize()

        self._index = index
        self._device_manager = peak.DeviceManager.Instance()

        self._cam: Optional[peak.Device] = None
        self._remote_device: Optional[peak.RemoteDevice] = None
        self._remote_map: Optional[peak.NodeMap] = None
        self._datastream: Optional[peak.DataStream] = None

        self._continuous: bool = False
        self._latest_frame: Optional[np.ndarray] = None

    def _open(self) -> None:
        self._device_manager.Update()
        devices = self._device_manager.Devices()

        if devices.empty():
            raise CameraError("No IDS Peak camera detected")

        self._cam = devices[self._index].OpenDevice(peak.DeviceAccessType_Control)

        self._remote_device = self._cam.RemoteDevice()
        self._remote_map = self._remote_device.NodeMaps()[0]

        # -------------------------------------------------------------
        # Ensure acquisition is STOPPED (CRITICAL)
        # -------------------------------------------------------------
        acq_stop = self._remote_map.FindNode("AcquisitionStop")
        if acq_stop and acq_stop.IsWriteable():
            acq_stop.Execute()
            acq_stop.WaitUntilDone()
            

        # -------------------------------------------------------------
        # Load default UserSet
        # -------------------------------------------------------------
        userset = self._find_node("UserSetSelector")
        if userset.IsWriteable():
            userset.SetCurrentEntry("Default")
            self._find_node("UserSetLoad").Execute()
            self._find_node("UserSetLoad").WaitUntilDone()

        # -------------------------------------------------------------
        # Disable auto features (CORRECT ORDER)
        # -------------------------------------------------------------
        exposure_auto = self._find_node("ExposureAuto")
        if exposure_auto.IsAvailable() and exposure_auto.IsWriteable():
            exposure_auto.SetCurrentEntry("Off")

        gain_auto = self._find_node("GainAuto")
        if gain_auto.IsAvailable() and gain_auto.IsWriteable():
            gain_auto.SetCurrentEntry("Off")

        # -------------------------------------------------------------
        # Full-frame watch window
        # -------------------------------------------------------------
        width = self._find_node("Width").Maximum()
        height = self._find_node("Height").Maximum()
        self.watch_window = (0, width, 0, height)

        self._continuous = False
        self._latest_frame = None

    def _close(self) -> None:
        if self._continuous:
            self._do_stop_stream()

        del self.cam

        peak.Library.Close()

    # -----------------------------------------------------------------
    # Typed node access (authoritative)
    # -----------------------------------------------------------------

    @overload
    def _find_node(
        self, key: Literal["ExposureTime", "Gain", "AcquisitionFrameRate"]
    ) -> peak.FloatNode: ...

    @overload
    def _find_node(
        self, key: Literal["OffsetX", "OffsetY", "Width", "Height", "TLParamsLocked"]
    ) -> peak.IntegerNode: ...

    @overload
    def _find_node(
        self,
        key: Literal["ExposureAuto", "ExposureMode", "GainSelector", "UserSetSelector"],
    ) -> peak.EnumerationNode: ...

    @overload
    def _find_node(
        self, key: Literal["UserSetLoad", "AcquisitionStart", "AcquisitionStop"]
    ) -> peak.CommandNode: ...

    @overload
    def _find_node(self, key: str) -> peak.Node: ...

    def _find_node(self, key: str) -> peak.Node:
        if not self._remote_map:
            raise InvalidStateError("Camera not open")

        node = self._remote_map.FindNode(key)

        expected_cls = _NODE_REGISTRY.get(key)
        runtime_cls = _NODETYPE_TO_CLASS.get(node.Type())

        if expected_cls and runtime_cls and runtime_cls is not expected_cls:
            raise CameraError(
                f"Node '{key}' expected {expected_cls.__name__}, "
                f"but device reports {runtime_cls.__name__}"
            )

        return node

    # -----------------------------------------------------------------
    # Parameter handling
    # -----------------------------------------------------------------

    def _set(self, parameter: Parameter, value: CameraValue) -> bool:
        was_streaming = self._continuous

        try:
            if was_streaming:
                self._do_stop_stream()

            if parameter == Parameter.EXPOSURE:
                self._find_node("ExposureTime").SetValue(float(value))
                return True

            if parameter == Parameter.GAIN:
                self._find_node("Gain").SetValue(float(value))
                return True

            return False

        except Exception:
            return False

        finally:
            if was_streaming:
                try:
                    self._do_start_stream()
                except Exception:
                    pass 


    # -----------------------------------------------------------------
    # Watch window (ROI)
    # -----------------------------------------------------------------


    def _apply_roi_constraints(self, x, y, w, h):
        offx = self._find_node("OffsetX")
        offy = self._find_node("OffsetY")
        width = self._find_node("Width")
        height = self._find_node("Height")

        x = (x // offx.Increment()) * offx.Increment()
        y = (y // offy.Increment()) * offy.Increment()
        w = (w // width.Increment()) * width.Increment()
        h = (h // height.Increment()) * height.Increment()

        return x, y, w, h


    def _watch(self, x_off: int, width: int, y_off: int, height: int) -> bool:
        try:
            was_streaming = self._continuous
            if was_streaming:
                self._do_stop_stream()

            x, y, w, h = self._apply_roi_constraints(x_off, y_off, width, height)
            self._find_node("Width").SetValue(w)
            self._find_node("Height").SetValue(h)
            self._find_node("OffsetX").SetValue(x)
            self._find_node("OffsetY").SetValue(y)

            self.watch_window = (x_off, width, y_off, height)
            return True

        except Exception as e:
            print("ROI set failed:", e)
            return False

        finally:
            if was_streaming:
                self._do_start_stream()


    def _reset_watch_window(self) -> None:
        if not self._remote_map:
            return
        w = self._find_node("Width").Maximum()
        h = self._find_node("Height").Maximum()
        self._watch(0, w, 0, h)

    # -----------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------

    def _prepare_datastream(self) -> None:
        self._datastream = self._cam.DataStreams()[0].OpenDataStream()
        payload = self._remote_map.FindNode("PayloadSize").Value()
        min_buffers = self._datastream.NumBuffersAnnouncedMinRequired()

        for _ in range(min_buffers * 5):
            buf = self._datastream.AllocAndAnnounceBuffer(payload)
            self._datastream.QueueBuffer(buf)

    def _do_start_stream(self) -> None:
        if not self._datastream:
            self._prepare_datastream()

        self._find_node("TLParamsLocked").SetValue(1)
        self._datastream.StartAcquisition()
        self._find_node("AcquisitionStart").Execute()
        self._find_node("AcquisitionStart").WaitUntilDone()
        self._continuous = True

    def _do_stop_stream(self) -> None:
        if not self._continuous:
            return

        self._find_node("AcquisitionStop").Execute()
        self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
        self._datastream.Flush(peak.DataStreamFlushMode_DiscardAll)
        self._find_node("TLParamsLocked").SetValue(0)
        self._continuous = False

    def _read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._continuous:
            self._do_start_stream()

        try:
            buffer = self._datastream.WaitForFinishedBuffer(1000)
            img = peak_ipl.Image.CreateFromSizeAndBuffer(
                buffer.PixelFormat(),
                buffer.BasePtr(),
                buffer.Size(),
                buffer.Width(),
                buffer.Height(),
            ).get_numpy_2D()

            self._datastream.QueueBuffer(buffer)

            frame = cv2.cvtColor(
                np.ascontiguousarray(img, dtype=np.uint8),
                cv2.COLOR_GRAY2BGR,
            )

            self._latest_frame = frame
            return True, frame

        except Exception:
            return False, None
