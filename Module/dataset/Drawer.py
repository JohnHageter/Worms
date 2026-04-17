import cv2
import numpy as np
import os


class ROIDrawer:
    def __init__(self, frame, initial_radius=50):
        self.frame = frame
        self.radius = initial_radius
        self.cursor = (0, 0)
        self.wells = []
        self.preview_frame = frame.copy()

    def mouse_callback(self, event, x, y, flags, param):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Confirm well placement
            self.wells.append((x, y, self.radius))

    def draw(self):
        cv2.namedWindow("Draw Wells")
        cv2.setMouseCallback("Draw Wells", self.mouse_callback)

        while True:
            self.preview_frame[:] = self.frame.copy()
            # Draw confirmed wells
            for w in self.wells:
                cv2.circle(self.preview_frame, (w[0], w[1]), w[2], (0, 255, 0), 2)
            # Draw preview well at cursor
            cv2.circle(self.preview_frame, self.cursor, self.radius, (0, 0, 255), 1)

            cv2.imshow("Draw Wells", self.preview_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC → finish
                break
            elif key == ord("+") or key == ord("="):
                self.radius += 10
            elif key == ord("-") and self.radius > 1:
                self.radius -= 10

        cv2.destroyWindow("Draw Wells")
        return self.wells

    def save(self, filepath):
        """
        Save the wells to a .npy file.
        Each well is (x, y, radius)
        """
        np.save(filepath, np.array(self.wells))
        print(f"Saved {len(self.wells)} wells to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load wells from a .npy file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")
        wells = np.load(filepath)
        return wells.tolist()


class RectROIDrawer:
    """
    Draw rectangular ROIs by click-drag.
    Hold CTRL while dragging to lock to a square.
    """

    def __init__(self, frame):
        self.frame = frame
        self.preview_frame = frame.copy()

        self.rois = []

        self.dragging = False
        self.start_pt = None
        self.end_pt = None
        self.ctrl_pressed = False

    # ---------------- Mouse Callback ----------------
    def mouse_callback(self, event, x, y, flags, param):
        self.ctrl_pressed = bool(flags & cv2.EVENT_FLAG_CTRLKEY)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end_pt = (x, y)

            x0, y0, w, h = self._compute_rect()
            if w > 0 and h > 0:
                self.rois.append((x0, y0, w, h))

            self.start_pt = None
            self.end_pt = None

    # ---------------- Geometry ----------------
    def _compute_rect(self):
        if self.start_pt is None or self.end_pt is None:
            return None

        x1, y1 = self.start_pt
        x2, y2 = self.end_pt

        dx = x2 - x1
        dy = y2 - y1

        if self.ctrl_pressed:
            side = max(abs(dx), abs(dy))
            dx = np.sign(dx) * side
            dy = np.sign(dy) * side

        x0 = int(min(x1, x1 + dx))
        y0 = int(min(y1, y1 + dy))
        w = int(abs(dx))
        h = int(abs(dy))

        return x0, y0, w, h

    # ---------------- Main Loop ----------------
    def draw(self):
        cv2.namedWindow("Draw ROIs")
        cv2.setMouseCallback("Draw ROIs", self.mouse_callback)

        while True:
            self.preview_frame[:] = self.frame.copy()

            # Draw confirmed ROIs
            for x, y, w, h in self.rois:
                cv2.rectangle(
                    self.preview_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

            # Draw preview ROI
            if self.dragging:
                rect = self._compute_rect()
                if rect is not None:
                    x0, y0, w, h = rect
                    color = (255, 0, 0) if self.ctrl_pressed else (0, 0, 255)
                    cv2.rectangle(
                        self.preview_frame,
                        (x0, y0),
                        (x0 + w, y0 + h),
                        color,
                        1,
                    )

            cv2.imshow("Draw ROIs", self.preview_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                break

        cv2.destroyWindow("Draw ROIs")
        return self.rois

    # ---------------- Persistence ----------------
    def save(self, filepath):
        """
        Save ROIs to a .npy file.
        Each ROI is (x, y, w, h)
        """
        np.save(filepath, np.array(self.rois))
        print(f"Saved {len(self.rois)} ROIs to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load ROIs from a .npy file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")
        return np.load(filepath).tolist()
