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
                self.radius += 1
            elif key == ord("-") and self.radius > 1:
                self.radius -= 1

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
