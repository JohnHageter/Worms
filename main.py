from Module.io.OpenCVCamera import OpenCVCamera
import cv2

cap = OpenCVCamera()
cap.open()


cap.start_stream(fps=.01)

while True:
    frame = cap.grab_last_frame()
    if frame is None:
        # print(f"{fail_count} Snapshot failed.")
        continue

    cv2.imshow("Live", frame)
    frame = None

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.stop_stream()
cap.close()
cv2.destroyAllWindows()
