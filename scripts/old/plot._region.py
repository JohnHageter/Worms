


import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from matplotlib.lines import Line2D

DATA_DIR = Path("D:/Tracking_Data/Sachi/T3/week_1_out_v3")
WELL_FILES = sorted(DATA_DIR.glob("well_*.h5"))

WELL_DIAMETER_MM = 35.0
FPS = 2.0 
BIN_FRAMES = 1000  
SMOOTH_SIGMA = 3  
FRAMES_PER_VIDEO = int(FPS * 3530)

REGION_CORE = 1
REGION_WALL = 2

REGION_COLORS = {
    REGION_CORE: "green",
    REGION_WALL: "orange",
}


from collections import Counter


for path in WELL_FILES:
    with h5py.File(path, "r") as h5:
        print(h5["tracking"].shape)
        main = h5["main"][:] if "main" in h5 else h5["tracking"][:]
    region_names = {
        0: "core",
        2: "wall",
    }

    for path in WELL_FILES:
        with h5py.File(path, "r") as h5:
            main = h5["main"][:] if "main" in h5 else h5["tracking"][:]

        region = main["region"].astype(int)

        counts = Counter(region)

        labels = []
        sizes = []

        for k, v in counts.items():
            labels.append(region_names.get(k, str(k)))
            sizes.append(v)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=[REGION_COLORS.get(k, "gray") for k in counts.keys()],
        )

        ax.set_title(path.name)

        plt.tight_layout()
        plt.show()
