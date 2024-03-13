# Description: This script is used to process images of a light table and identify
# particles in subsequent images. It is used to measure the size of particles and ultimately
# to count the number of particles in each image for sediment transport studies.
# 2024-03-10: Tobias Mueller, initial version

import logging
from pathlib import Path
from datetime import datetime
import time

# database for storing results
import sqlite3

# image processing
import cv2
import numpy as np
import skimage

# from skimage.transform import warp
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk

import pandas as pd

# plotting
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class ImageLooperDOF:
    def __init__(self, c):
        self.images = list(Path(c["images"]["path"]).rglob("*.tif"))

        self.file_background = Path(c["images"]["file_background"])
        self.debug = c["debugging"]["debug"] == "True"
        self.c = c

        self.db_file = Path(
            c["output"]["path"],
            f'{c["config"]["run_name"]}{c["output"]["file_db_append"]}',
        )
        # create database
        self.create_db(self.db_file)

        # load calibration image
        self.img_cal = self.load_image_gray(
            Path(c["images"]["file_calibration"]), crop=False
        )

        # TODO: use calibration image to calculate pixel to mm ratio

        self.csv_file = f'{c["config"]["run_name"]}{c["output"]["file_csv_append"]}'

        # set up logging, will be one file per execution
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(__file__).parent.parent / "logs" / f"{datetime.now()}.log"
                ),
            ],
        )

    def create_db(self, db_file):

        db = sqlite3.connect(db_file)

        # clear database
        db.execute("DROP TABLE IF EXISTS particles")
        db.execute("DROP TABLE IF EXISTS images")
        db.execute("DROP TABLE IF EXISTS seconds")

        # create table to append particle recognitions
        db.execute(
            "CREATE TABLE particles (id INTEGER PRIMARY KEY, image TEXT, time REAL, x REAL, y REAL, width REAL, height REAL, area REAL)"
        )
        # create table to append per image data
        db.execute(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, image TEXT, time REAL, particles INTEGER)"
        )
        # create table to append per second data
        db.execute(
            "CREATE TABLE seconds (id INTEGER PRIMARY KEY, time REAL, particles INTEGER)"
        )

        # close database
        db.commit()
        db.close()

    def load_image_gray(self, path, crop=False):
        # load image and convert to grayscale
        img = cv2.imread(path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.noise_filter(img)
        img = self.sharpen_image(img)

        # crop image to background-image area
        if crop:
            crop_left = int(self.c["images"]["crop_left"])
            crop_right = int(self.c["images"]["crop_right"])
            crop_top = int(self.c["images"]["crop_top"])
            crop_bottom = int(self.c["images"]["crop_bottom"])
            img = img[
                crop_top : img.shape[0] - crop_bottom,
                crop_left : img.shape[1] - crop_right,
            ]

        return img

    def noise_filter(self, img, pixels=3):
        # filter out noise
        img = cv2.GaussianBlur(img, (pixels, pixels), 0)
        img = cv2.medianBlur(img, pixels)
        return img

    def sharpen_image(self, img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img

    # threshold image to isolate particles
    def threshold_image(self, img):
        # threshold image to isolate particles
        # TODO: make threshold value configurable
        img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)[1]
        # TODO: make threshold value configurable
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # TODO: make threshold value configurable
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        return img

    def write_to_sqlite(self, db_file, image_path, particles):

        # open database
        db = sqlite3.connect(db_file)

        # add image data to sqlite database
        db.execute(
            "INSERT INTO images (image, time, particles) VALUES (?, ?, ?)",
            (image_path.as_posix(), float(image_path.stem), None),
        )

        for particle in particles:
            # add to sqlite database
            db.execute(
                "INSERT INTO particles (image, time, x, y, width, height, area) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    image_path.as_posix(),
                    float(image_path.stem),
                    particle.bbox[1],
                    particle.bbox[0],
                    particle.bbox[3] - particle.bbox[1],
                    particle.bbox[2] - particle.bbox[0],
                    particle.area,
                ),
            )

        # close database
        db.commit()
        db.close()

    def analyze_image(self, db_file, image_path, img_bg):
        particles = []

        # load image
        img = self.load_image_gray(image_path, crop=True)

        # invert grayscale and stretch greyscale between 0 and 255
        # img = cv2.bitwise_not(img)
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # threshold image to isolate particles
        # img = self.threshold_image(img)

        return img

    def draw_flow(self, img, flow, step=4):

        h, w = img.shape[:2]
        y, x = (
            np.mgrid[step / 2 : h : step, step / 2 : w : step]
            .reshape(2, -1)
            .astype(int)
        )
        #  filter out flow with small maginitude
        fx, fy = flow[y, x].T
        flow_mag = np.sqrt(fx**2 + fy**2)

        fx, fy = fx[flow_mag > 1], fy[flow_mag > 1]
        x, y = x[flow_mag > 1], y[flow_mag > 1]

        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        return vis

    def run(self):
        frame_count = 0
        frame_count_total = 0
        tic = time.perf_counter()

        images = sorted(self.images)

        # load background image
        img_bg = self.load_image_gray(self.file_background, crop=False)
        img_prev = None
        df_part_prev = None

        # loop through images
        for image_path in images:

            if img_prev is None:
                img_prev = self.load_image_gray(image_path, crop=True)
                continue

            img = self.analyze_image(self.db_file, image_path, img_bg)

            if self.debug:

                lk_params = dict(
                    winSize=(200, 200),
                    maxLevel=10,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        10,
                        0.1,
                    ),
                )

                feature_params = dict(
                    maxCorners=100, qualityLevel=0.3, minDistance=100, blockSize=15
                )

                if img_prev is not None:

                    # opencv dense optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        img_prev, img, None, 0.5, 5, 12, 3, 5, 1.2, 0
                    )

                    # show img before and after
                    cv2.imshow("flow", self.draw_flow(img, flow))

                    key = cv2.waitKey(1000)

                key = cv2.waitKey(200)
                if key == 27:  # if ESC is pressed, exit loop
                    cv2.destroyAllWindows()

            img_prev = img
