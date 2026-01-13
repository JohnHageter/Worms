class CameraError(Exception):
    """Base class for all camera-related errors."""


class InvalidStateError(CameraError):
    """Invalid operation for the current camera state."""


class CaptureError(CameraError):
    """Frame capture failed."""
