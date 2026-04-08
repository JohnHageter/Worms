## Worm Tracker
This is a software designed at captuing a timelapse based acquisition of planaria in multi well plates to track movement and fissioning events.

This software comes in two pieces:
1. The acuisition script which is designed to capture images over a period of time using a IDS uEYE campera
2. The geometry based keypoint tracking which tracks worms movement in the well plates


### Acquisition script
1. ...
2.
3.


### Tracking script
1. ...
2.
3.


dataset output structure
well_i.h5
├── /main
│   ├── centroid_x [T]
│   ├── centroid_y [T]
│   ├── head_x     [T]
│   ├── head_y     [T]
│   ├── tail_x     [T]
│   ├── tail_y     [T]
│   └── region     [T]
└── /fission
    ├── track_id   [N x T]
    ├── centroid_x [N x T]
    ├── centroid_y [N x T]
    └── region     [N x T]