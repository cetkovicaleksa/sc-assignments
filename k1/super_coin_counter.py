# sv77/2022 Алекса Ћетковић
#%%
_echo: bool = False

import sys
from pathlib import Path
from functools import reduce
from itertools import chain, cycle

import numpy as np
import pandas as pd
import cv2

# if _echo:
#     # import matplotlib.pyplot as plt
#     #%matplotlib inline
#     pass

#%%
def imshow(img, cvt: int | None = None, *args, **kwargs):
    if _echo:
        import matplotlib.pyplot as plt

        if cvt is not None:
            img = cv2.cvtColor(img, code=cvt)

        plt.imshow(img, *args, **kwargs)
        plt.axis(kwargs.get('axis', 'off'))
        plt.tight_layout()
        plt.show()

def histshow(*channels, img = None, colours = None):
    if _echo:
        import matplotlib.pyplot as plt

        colours = cycle(colours or ["blue","green","red","orange","purple","cyan","magenta","yellow",])
        plt.figure(figsize=(8, 5))

        for ch, colour in zip(cv2.split(img) if img is not None else channels, colours):
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
            plt.plot(hist, color=colour)
            plt.fill_between(range(256), hist.flatten(), color=colour, alpha=0.4)
        
        plt.show()

#%%
GOLD_COIN_VAL = 1
RED_COIN_VAL = 2
STAR_COIN_VAL = 5

def evaluate_image(bgr) -> int:
    bgr = cv2.resize(bgr, (1_920, 1_080))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    # (0 - 179), (0 - 255), (0 - 255)
    H, S, V = cv2.split(hsv)
    # imshow(hsv, cv2.COLOR_HSV2RGB)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5, 5))
    hsv = cv2.merge([H, S, (V := clahe.apply(V))])

    s_thresh = np.percentile(S, 80)
    s_thresh_low = np.percentile(S, 45)
    color_ranges = [
        # [(0, 200, 0), (255, 255, 255)], # TODO
        [(0//2, s_thresh, 100), (15//2, 255, 255)],
        [(30//2, s_thresh, 150), (50//2, 255, 255)],
        [(50//2, s_thresh_low, 100), (60//2, 255, 255)],
    ]
    color_mask = reduce(
        cv2.bitwise_or,
        (
            cv2.inRange(hsv, lower, upper) 
                for lower, upper in color_ranges
        )
    )

    # imshow(color_mask, cmap='gray')

    # idea to use elipse kernel from: <https://www.github.com/Luke-Byrne-MEng/UK-Coin-Counter>
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel3, iterations=3)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

    # V = cv2.bitwise_and(V, V, mask=color_mask)
    # _, mask = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.adaptiveThreshold(
        V, 
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=27, 
        C=2
    )

    mask = cv2.bitwise_and(mask, mask, mask=color_mask)

    # dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0)
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(mask, sure_fg)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_contours = []
    coins = 0
    red_coins = 0
    star_coins = 0

    for cnt in contours:
        if not is_coin(cnt):
            continue

        coin_contours.append(cnt)
        coins += 1
        
        if is_red_coin(cnt, hsv):
            red_coins += 1
        elif is_star_coin(cnt):
            star_coins += 1

    if _echo:
        bgr_anot = bgr.copy()
        cv2.drawContours(bgr_anot, coin_contours, -1, (255, 0, 255), 3)
        imshow(mask, cmap='gray')
        imshow(bgr_anot, cv2.COLOR_BGR2RGB)
    
    return coins * GOLD_COIN_VAL + (RED_COIN_VAL - GOLD_COIN_VAL) * red_coins + (STAR_COIN_VAL - GOLD_COIN_VAL) * star_coins
    
# some logic from classifying contours from: <https://www.chat.openai.com>
def is_coin(contour, min_area=800, max_area=20_000, min_circularity=.4):
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / perimeter ** 2
    if circularity < min_circularity:
        return False
    
    return True

def is_red_coin(coin_contour, hsv, redness=.4):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [coin_contour], -1, 255, thickness=cv2.FILLED)

    H, S, _ = cv2.split(hsv)
    H, S = H[mask == 255], S[mask == 255]

    if len(H) == 0:
        return False
    
    red_mask = ((H < 10) | (H > 170)) & (S > 120)
    red_ratio = np.mean(red_mask.astype(np.uint8))
    return red_ratio > redness

def is_star_coin(coin_contour):
    area = cv2.contourArea(coin_contour)
    return area > 5_000

#%%
# img = cv2.imread(images[0])
# evaluate_image(img)

#%%
if __name__ == '__main__':
    data: Path

    if len(sys.argv) <= 1 or not (data := Path(sys.argv[1])).is_dir():
        data = Path('data')

    df = pd.read_csv(data/"coin_value_count.csv")
    images, values = df['image_name'].map(lambda name: data/name).to_numpy(), \
                     df['coins_value'].to_numpy(dtype=np.uint8)
    
#%%
    pred = np.array(
        [evaluate_image(img) for img in map(cv2.imread, images)],
        dtype=np.uint8
    )

    mse = np.mean((pred - values) ** 2)
    print(mse)

    if _echo:
        for img, p, a in zip(images, pred, values):
            print(f"{img.name}: [predicted={p}, actual={a}]")

        assert False, "Change this"

