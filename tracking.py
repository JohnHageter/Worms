import cv2
import numpy as np
import h5py
from pathlib import Path
from Module.utils import *
from Module.Dataset import *
from Module.ImageProcessor import *
from Module.Tracker import *


data_folder = Path("Data/timelapse_images_2")
dataset_file = data_folder / "dataset.h5"

length_threshold = 300  
min_area = 20
max_area = 150
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10    


if not dataset_file.exists():
    print("Generating H5 dataset...")
    generate_dataset(data_folder, output_file=dataset_file)

with h5py.File(dataset_file, 'r') as f:
    image_data = f['images/data'][:]
    timestamps = f['images/timestamps'][:]

wells, masks, parameters = detect_wells(
    image_data[137],
    hough_param1=25,
    hough_param2=24,
    well_diameter_mm=35.0,
    mm_per_pixel=0.187,
    well_tolerance=0.15,
    min_radius=96,
    max_radius=129,
    n_wells=24
)


out_path = data_folder / 'output'
out_path.mkdir(exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))

for id, well in enumerate(wells):

    first_frame = crop_well(image_data[0], well)
    h, w = first_frame.shape

    writer = cv2.VideoWriter(
        str(out_path / f'well_{id:02d}.mp4'),
        fourcc,
        fps,
        (w, h),
        isColor=True
    )

    bg_frame = clahe.apply(image_data[137])
    well_bg = filter_homomorphic(snap_background(crop_well(bg_frame, well), 11, 7))
    mask = crop_well(masks[id], well).astype(np.uint8) * 255

    bg_thresh = None

    for i in range(image_data.shape[0]):

        well_frame = crop_well(image_data[i], well)
        well_frame = filter_homomorphic(well_frame, gamma_h=1.2)

        fg = cv2.subtract(well_bg, well_frame)
        fg = cv2.bitwise_and(fg, fg, mask=mask)

        _, thresh = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        )

        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.erode(thresh,kernel, iterations=1)

        if i == 136:
            bg_thresh = thresh.copy()

        if bg_thresh is not None:
            thresh = cv2.subtract(thresh, bg_thresh)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8
        )

        vis = cv2.cvtColor(well_frame, cv2.COLOR_GRAY2BGR)



        cv2.imshow("Thesholding", thresh)
        cv2.imshow("Write", vis)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    #writer.release()

cv2.destroyAllWindows()