import cv2
import numpy as np
import os
from pathlib import Path
import h5py
from datetime import datetime
from Module.utils import read_timestamp

'''
Dataset should move to a compressed mp4 with h5 metadata
'''
def open_dataset(video_path):
    if not Path(video_path).exists():
        raise ValueError(f"Video file {video_path} does not exist.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}.")

    return cap

def crop_well(image, well):
    H, W = image.shape[:2]
    x,y,r = well

    r = int(round(r))
    x = int(round(x))
    y = int(round(y))
    size = 2 * r

    cropped = np.zeros((size, size), dtype=image.dtype)


    x1_src = max(x - r, 0)
    x2_src = min(x + r, W)
    y1_src = max(y - r, 0)
    y2_src = min(y + r, H)

    x1_dst = max(0, r - x)
    y1_dst = max(0, r - y)
    x2_dst = x1_dst + (x2_src - x1_src)
    y2_dst = y1_dst + (y2_src - y1_src)

    cropped[y1_dst:y2_dst, x1_dst:x2_dst] = image[y1_src:y2_src, x1_src:x2_src]

    return cropped

def _sort_wells(wells, well_masks, row_tol=50):
    """
    Sort wells top-left to bottom-right.

    wells: list of (x, y, r)
    well_masks: list of corresponding masks
    row_tol: pixels to group wells in the same row
    """

    combined = list(zip(wells, well_masks))
    combined = sorted(combined, key=lambda wm: wm[0][1])  # sort by y first

    sorted_rows = []
    current_row = []
    current_y = None

    for well, mask in combined:
        y = well[1]
        if current_y is None or abs(y - current_y) <= row_tol:
            current_row.append((well, mask))
            current_y = y if current_y is None else (current_y + y)/2
        else:
            # sort the current row by x (left to right)
            current_row.sort(key=lambda wm: wm[0][0])
            sorted_rows.extend(current_row)
            current_row = [(well, mask)]
            current_y = y
    if current_row:
        current_row.sort(key=lambda wm: wm[0][0])
        sorted_rows.extend(current_row)

    wells_sorted, masks_sorted = zip(*sorted_rows)
    return list(wells_sorted), list(masks_sorted)

@DeprecationWarning
def get_image_paths(folder):
    type = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in type
    ]

    return sorted(files)

# Unsure of final frame rate. Barring num of frames can range from 5000 to 1000000 
# only work with individual frames at a time instead of loading all frames at once.    
@DeprecationWarning
def load_frame(image_path, scale=1.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def detect_wells(
    frame,
    *,
    well_diameter_mm=35.0,
    mm_per_pixel=0.18,
    well_tolerance=0.1,
    n_wells=24,
    hough_dp=1.5,
    min_radius=None,
    max_radius=None,
    min_dist_factor=1.05,
    hough_param1=100,
    hough_param2=30,
    debug=False
):
    
    if frame is None:
        raise ValueError("No image provided.")

    gray = frame.copy()
    gray_blur = cv2.medianBlur(gray, 7)
    H, W = gray.shape

    diameter_px = well_diameter_mm / mm_per_pixel
    if min_radius is None or max_radius is None:
        radius_px   = diameter_px / 2
        min_radius  = int(radius_px * (1 - well_tolerance))
        max_radius  = int(radius_px * (1 + well_tolerance))
    
    min_dist    = int(diameter_px * min_dist_factor)

    def run_hough(param1, param2, rmin, rmax):
        parameters = {
            "well_diameter_mm": well_diameter_mm,
            "mm_per_pixel": mm_per_pixel,
            "well_tolerance": well_tolerance,
            "n_wells": n_wells,
            "hough_dp": hough_dp,
            "min_dist_factor": min_dist_factor,
            "hough_param1": param1,
            "hough_param2": param2,
            "min_radius": rmin,
            "max_radius": rmax,
        }

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=hough_dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=rmin,
            maxRadius=rmax,
        )

        wells = []
        masks = []

        if circles is None:
            print("No wells detected.")
            return wells, masks

        for x, y, r in np.round(circles[0]).astype(int):
            if r <= 0 or x < 0 or y < 0 or x >= W or y >= H:
                continue

            wells.append((float(x), float(y), float(r)))
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            masks.append(mask.astype(bool))

        # enforce equal size + N
        if wells and n_wells is not None:
            rs = np.array([r for _, _, r in wells])
            r_med = np.median(rs)
            keep = np.abs(rs - r_med) / r_med < well_tolerance

            wells = [w for w, k in zip(wells, keep) if k]
            masks = [m for m, k in zip(masks, keep) if k]

            wells = wells[:n_wells]
            masks = masks[:n_wells]
            wells, masks = _sort_wells(wells, masks)

        return wells,masks, parameters

    if debug:
        def nothing(x): pass

        cv2.namedWindow("Hough Debug")
        cv2.createTrackbar("param1", "Hough Debug", hough_param1, 50, nothing)
        cv2.createTrackbar("param2", "Hough Debug", hough_param2, 80, nothing)
        cv2.createTrackbar("minRadius", "Hough Debug", min_radius, max_radius * 2, nothing)
        cv2.createTrackbar("maxRadius", "Hough Debug", max_radius, max_radius * 2, nothing)

        while True:
            p1 = max(1, cv2.getTrackbarPos("param1", "Hough Debug"))
            p2 = max(1, cv2.getTrackbarPos("param2", "Hough Debug"))
            rmin = cv2.getTrackbarPos("minRadius", "Hough Debug")
            rmax = cv2.getTrackbarPos("maxRadius", "Hough Debug")

            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            wells_tmp, masks_tmp, _ = run_hough(p1, p2, rmin, rmax)

            for x, y, r in wells_tmp:
                cv2.circle(disp, (int(x), int(y)), int(r), (0, 255, 0), 2)

            cv2.putText(
                disp,
                f"Diameter = {diameter_px:.1f}px (+/-{well_tolerance*100:.0f}%)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            disp = cv2.resize(disp, (640, 512), interpolation=cv2.INTER_AREA)
            cv2.imshow("Hough Debug", disp)

            if cv2.waitKey(100) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        print(f'Hough parameters: param1={p1}, param2={p2}, minRadius={rmin}, maxRadius={rmax}')
        wells, masks, parameters = run_hough(
            p1, p2, rmin, rmax
        )
        return wells, masks, parameters


    return run_hough(hough_param1, hough_param2, min_radius, max_radius)

def expand_wells(wells, scale=1.2):
    expanded = []
    for x, y, r in wells:
        r_new = r * scale
        expanded.append((x, y, r_new))
    return expanded

@DeprecationWarning
def generate_dataset(folder_path, output_file='dataset.h5', compression='gzip',compression_level=4):
    images = os.listdir(folder_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    images = [Path(folder_path) / img for img in images]
    
    first = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    H, W = first.shape
    N = len(images)

    with h5py.File(output_file, 'w') as h5f:
        dset = h5f.create_dataset(
            'images/data',
            shape=(N, H, W),
            dtype=np.uint8,
            chunks=(1, H, W),
            compression=compression,
            compression_opts=compression_level
        )

        ts = h5f.create_dataset(
            'images/timestamps',
            shape=(N,),
            dtype='f8',
        )

        h5f.create_dataset('/meta/source_folder', data=str(folder_path))
        h5f.create_dataset('/meta/creation_date', data=datetime.now().isoformat())

        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img.shape != (H, W):
                raise ValueError(f"Image {img_path} has different dimensions.")
            dset[i, :, :] = img
            dt = read_timestamp(str(img_path))
            ts[i] = dt.timestamp()

            if (i + 1) % 100 == 0 or (i + 1) == N:
                print(f'Processed {i + 1}/{N} images.')

        print(f'Dataset saved to {output_file}')



