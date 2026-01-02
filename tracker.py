import cv2
import numpy as np
from pathlib import Path
from Module.Worms import WormTracker
from Module.utils import *
from Module.Dataset import *
from Module.ImageProcessor import *
from Module.Tracker import *
import csv



data_folder = Path("Data/timelapse_images_2.1")
image_data = get_image_paths(data_folder)
first_frame = load_frame(image_path=image_data[0])

background = sample_background(image_data, n_frames=200)
background = background.astype(np.uint8)

wells, masks, parameters = detect_wells(
    first_frame,
    hough_param1=25,
    hough_param2=24,
    well_diameter_mm=35.0,
    mm_per_pixel=0.187,
    well_tolerance=0.15,
    min_radius=96,
    max_radius=129,
    n_wells=24
)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(
        Path('./outputs/tracked.mp4'),
        fourcc,
        5,
        (first_frame.shape[1], first_frame.shape[0]),
        isColor=True
    )

tracker = WormTracker(max_dist=300, max_missed=100)
track_records = []
for i, f in enumerate(image_data):
    img = load_frame(f)
    H,W = img.shape
    fg, thresh = extract_foreground(img, background, thresh_val=35)
    num_labels, labels, stats, centroids = find_components(thresh)
    worms = build_worms(labels, stats, centroids, wells, minimum_skeleton_area=100)
    worms_to_keep = filter_worms(worms, min_area=10)
    worm_tracks = tracker.update(worms_to_keep, frame_idx=i)
    #cv2.imshow("Threshold", cv2.resize(thresh, (W//2, H//2), cv2.INTER_AREA))



    visual = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for worm in worms_to_keep:
        cx, cy = map(int, worm['centroid'])
        end1 = tuple(map(int, worm['end1'])) if worm['end1'] is not None else None
        end2 = tuple(map(int, worm['end2'])) if worm['end2'] is not None else None


        cv2.circle(visual, (cx, cy), 3, (0,255,0), -1)
        
        if end1 is not None and end2 is not None:
            cv2.circle(visual, (end1[0], end1[1]), 3, (255,0,0), -1)
            cv2.circle(visual, (end2[0], end2[1]), 3, (0,0,255), -1)

    for well in wells:
        x, y, r = map(int, well)
        cv2.circle(visual, (x,y), r, (255,0,0), 1)

    for t in worm_tracks:
        cx ,cy = map(int, t.last_centroid)
        cv2.putText(visual, f'ID:{t.id}', (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    #cv2.imshow("Background", cv2.resize(background, (W//2,H//2), cv2.INTER_AREA))
    #cv2.imshow("Foreground", cv2.resize(fg, (W//2,H//2), cv2.INTER_AREA))
    #cv2.imshow("Video", cv2.resize(visual, (W,H), cv2.INTER_AREA))
    writer.write(visual)
    

    for t in worm_tracks:
        cx, cy = t.last_centroid
        e1 = t.last_end1
        e2 = t.last_end2

        track_records.append({
            "frame": i,
            "track_id": t.id,
            "well_id": t.well_id,
            "center_x": cx,
            "center_y": cy,
            "end1_x": e1[0] if e1 is not None else "",
            "end1_y": e1[1] if e1 is not None else "",
            "end2_x": e2[0] if e2 is not None else "",
            "end2_y": e2[1] if e2 is not None else "",
            "age": t.age,
            "missed": t.missed
        })




writer.release()
cv2.destroyAllWindows()

print(f"Saving tracking results to ./outputs/worm_tracks.csv")
output_csv = Path("./outputs/worm_tracks.csv")

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame", "track_id", "well_id",
            "center_x", "center_y",
            "end1_x", "end1_y",
            "end2_x", "end2_y",
            "age", "missed"
        ]
    )
    writer.writeheader()
    writer.writerows(track_records)