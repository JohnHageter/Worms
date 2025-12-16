import cv2
import numpy as np
import tifffile

def load_video(path):
    path_l = path.lower()
    if path_l.endswith((".tif", ".tiff")):
        stack = tifffile.imread(path)
        return stack, True
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError("Could not open video: " + path)
        return cap, False
    
def play_video(video, framerate, num_frames=10, shrink_factor=0.5):
    if isinstance(video, cv2.VideoCapture):
        for _ in range(num_frames):
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0,0), fx=shrink_factor, fy=shrink_factor)
            cv2.imshow("Video", frame)
            if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
                break
        video.release()
    else:
        for i in range(min(num_frames, video.shape[0])):
            frame = video[i]
            frame = cv2.resize(frame, (0,0), fx=shrink_factor, fy=shrink_factor)
            cv2.imshow("Video", frame)
            if cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def plot_well_masks(well_masks):
    import matplotlib.pyplot as plt
    num_wells = len(well_masks)
    cols = int(np.ceil(np.sqrt(num_wells)))
    rows = int(np.ceil(num_wells / cols))
    plt.figure(figsize=(15, 15))
    for i, mask in enumerate(well_masks):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title(f'Well {i}')
    plt.tight_layout()
    plt.show()