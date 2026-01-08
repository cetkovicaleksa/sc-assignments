"""sv77/2022 Алекса Ћетковић"""

# %% Imports
import sys
from pathlib import Path
from math import ceil, floor
from typing import cast
from contextlib import contextmanager

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

cv.ocl.setUseOpenCL(False)

# %% Utilities
# Pretty much the same, stripped down, utilities from here:
# <https://gist.github.com/v-goncharenko/d9e1fef1ee40fa36f610acebeabbcc25>  
#
# NOTE: I first tried to define a subclass of cv2::VideoCapture (that can be used with `with`), but that caused segmentation errors.

@contextmanager
def video_capture(path: str | Path, *args, **kwargs):

    video = cv.VideoCapture(str(path), *args, **kwargs)

    if not video.isOpened(): 
        raise IOError(f'Video {path} is not opened!')

    try:
        yield video
    finally:
        video.release()

def frames(video: cv.VideoCapture):
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break

        yield cast(np.ndarray, frame)

# %% Counting bones
def evaluate_video(path) -> int:
    eaten = 0
    frames_no_circle = float('inf') # count

    with video_capture(path) as video:
        for frame in frames(video):
            h, w, _ = frame.shape

            hitbox_size = ceil(.47 * min(h, w))
            hitbox_center = np.array([w // 2, h - hitbox_size // 2 - floor(h * .07)])

            hitbox_start = hitbox_center - hitbox_size // 2
            hitbox_end = hitbox_center + hitbox_size // 2
            np.clip(hitbox_start, 0, [w, h], out=hitbox_start)
            np.clip(hitbox_end, 0, [w, h], out=hitbox_end)

            hitbox = frame[hitbox_start[1]:hitbox_end[1], hitbox_start[0]:hitbox_end[0]]

            gray = cv.cvtColor(hitbox, cv.COLOR_BGR2GRAY)
            blur = cv.medianBlur(gray, 5)
            _, bright = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)

            circles = cv.HoughCircles(
                bright,  
                cv.HOUGH_GRADIENT,
                dp=1.2,
                minDist=hitbox_size//2,
                param1=50,
                param2=25,  
                minRadius=int(hitbox_size*.2),
                maxRadius=int(hitbox_size*.69)
            )
            
            if circles is not None and len(circles[0]) > 0:
                if frames_no_circle > 8:
                    eaten += 1

                frames_no_circle = 0
            else:
                frames_no_circle += 1

    return eaten

# %% Data
if __name__ == '__main__':
    data = Path('data')

    if len(sys.argv) > 1 and Path(data_path := sys.argv[-1]).is_dir():
        data = Path(data_path)

    count_csv = next(data.glob("*.csv"))
    df = pd.read_csv(
        count_csv, 
        converters={
            'video': lambda name: (data / name).with_suffix('.mp4'),
            'count': int
        }
    )

    videos, counts_true = df[['video', 'count']].to_numpy().T

# %% Test
    counts_pred = [evaluate_video(video) for video in videos]
    loss: float = mean_absolute_error(counts_true, counts_pred)
    print(loss)
