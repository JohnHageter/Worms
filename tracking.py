import os
import cv2
import numpy as np
import time


def load_dataset(path, resize_to=None):
    images = []
    valid_extensions = {".png", ".tif", ".jpeg", ".bmp", ".tiff", ".jpg"}

    for file in os.listdir(path):
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_path = os.path.join(path, file)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if resize_to:
                img = cv2.resize(img, resize_to)

            images.append(img)

    return np.stack(images) if images else np.array([])

def play_video(image_stack, fps = 30):
    assert image_stack.ndim in (3,4)
    delay = 1.0 / fps

    cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)

    for f in image_stack:
        start = time.time()
        if f.ndim == 2:
            disp = f
        else:
            disp = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

        cv2.imshow("Playback",disp)
        
        elapsed = time.time() - start
        remaining = max(1, int((delay - elapsed) * 1000))

        if cv2.waitKey(remaining) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows("Playback")


def split_6_wells(frame):
    H, W = frame.shape
    w = W // 3
    h = H // 2

    wells = []
    wells.append(frame[0:h,     0:w])        # Well 1
    wells.append(frame[0:h,     w:2*w])      # Well 2
    wells.append(frame[0:h,     2*w:3*w])    # Well 3
    wells.append(frame[h:2*h,   0:w])        # Well 4
    wells.append(frame[h:2*h,   w:2*w])      # Well 5
    wells.append(frame[h:2*h,   2*w:3*w])    # Well 6

    return wells


def create_ui(window_name):
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Threshold", window_name, 40, 255, lambda x: None)
    cv2.createTrackbar("MinArea", window_name, 50, 5000, lambda x: None)
    cv2.createTrackbar("Blur", window_name, 3, 25, lambda x: None)
    cv2.createTrackbar("BG Rate (x1000)", window_name, 10, 1000, lambda x: None)


def get_ui_params(window_name):
    thresh = cv2.getTrackbarPos("Threshold", window_name)
    minarea = cv2.getTrackbarPos("MinArea", window_name)
    blur = cv2.getTrackbarPos("Blur", window_name)
    if blur % 2 == 0: blur += 1  # blur must be odd
    bg_rate = cv2.getTrackbarPos("BG Rate (x1000)", window_name) / 1000.0
    return thresh, minarea, blur, bg_rate


def track_planaria_from_stack(image_stack, output_csv="tracks.csv"):
    if len(image_stack) == 0:
        print("Empty image stack.")
        return

    # First frame
    frame = image_stack[0]
    wells = split_6_wells(frame)
    bg_models = [w.copy().astype(np.float32) for w in wells]

    window = "Tracking"
    create_ui(window)

    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "well", "x", "y"])

        H, W = frame.shape
        w = W // 3
        h = H // 2

        for frame_idx, frame_gray in enumerate(image_stack):
            vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            wells = split_6_wells(frame_gray)

            thresh, minarea, blur, bg_rate = get_ui_params(window)

            for i, well in enumerate(wells):
                cv2.accumulateWeighted(well, bg_models[i], bg_rate)
                bg = cv2.convertScaleAbs(bg_models[i])

                fg = cv2.absdiff(well, bg)
                if blur > 1:
                    fg = cv2.GaussianBlur(fg, (blur, blur), 0)

                _, bw = cv2.threshold(fg, thresh, 255, cv2.THRESH_BINARY)
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > minarea:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            global_x = cx + (i % 3) * w
                            global_y = cy + (i // 3) * h

                            cv2.circle(vis, (global_x, global_y), 5, (0,0,255), -1)
                            writer.writerow([frame_idx, i+1, global_x, global_y])

                cv2.rectangle(vis,
                              ((i % 3) * w, (i // 3) * h),
                              ((i % 3) * w + w, (i // 3) * h + h),
                              (0,255,0), 1)

            cv2.imshow(window, scale_fixed(vis, scale=0.35))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Tracking complete from image stack.")


def scale_fixed(img, scale=0.3):
    """Scale image down by a fixed factor (e.g., 0.3 = 30%)."""
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)



images = load_dataset(".//timelapse_images")
track_planaria_from_stack(images)
