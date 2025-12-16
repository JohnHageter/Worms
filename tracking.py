from Module.utils import *
from Module.ImageProcessor import *
from Module.Tracker import *
import matplotlib.pyplot as plt
from pprint import pprint

video_path = "./Data/Test_1.tif"

video, _ = load_video(video_path)
#play_video(video, framerate=30, num_frames=1000, shrink_factor=0.5)

wells,masks,test = detect_wells(video[0], debug=True)

print("Detected wells:")
print(wells)
well_masks = mask_wells(video[0], wells)

wells, well_masks_sorted = sort_wells(wells, well_masks)
print("Sorted wells:")
print(wells)

#plot_well_masks(well_masks_sorted)
background, fg_masks = subtract_background(video, n_bg=1, minimum=20, maximum=250, display=True)
fg_array = np.stack(fg_masks, axis=0)
print(fg_array.shape)
#play_video(fg_array, framerate=30, num_frames=1000, shrink_factor=0.5)

well_tracks, lineage = track_worms(video, well_masks_sorted, fg_array,
                                  min_area_for_head_tail=1000,
                                  max_tracking_dist=100,
                                  max_still_frames=10)

print("Tracking results:")
pprint(lineage)

visualize_tracking(video, well_tracks, fps=10)