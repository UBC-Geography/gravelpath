# Description: This script is used to process images of a light table and identify
# particles in subsequent images. It is used to measure the size of particles and ultimately
# to count the number of particles in each image for sediment transport studies.
# 2024-03-10: Tobias Mueller, initial version
# 2024-05-08: Sol Leader-cole, adjusted version

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


class ImageLooper:
    def __init__(self, c):
        self.images = list(Path(c["images"]["path"]).rglob("*.tif"))

        self.file_background = Path(c["images"]["background"])
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
            Path(c["images"]["calibration"]), crop=True
        )

        # TODO: use calibration image to calculate pixel to mm ratio
        # need to decide if this is needed, lighttable camera shouldn't change position too much between experiments, can measure manually before running
        # in alex's experiments dots are 45 pixels apart

        self.csv_file = f'{c["config"]["run_name"]}{c["output"]["file_csv_append"]}'

        #setting the pixel value threshold for particles to be detected
        self.bin_threshold = c['cost_parameters']['binary_threshold']

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
        db.execute("DROP TABLE IF EXISTS trajectories")

        # create table to append particle recognitions
        db.execute(
            "CREATE TABLE particles (id INTEGER PRIMARY KEY, image TEXT, time REAL, x REAL, y REAL, width REAL, height REAL, area REAL, eccentricity REAL)"
        )
        # create table to append per image data
        db.execute(
            "CREATE TABLE images (id INTEGER PRIMARY KEY, image TEXT, time REAL, particles INTEGER, bedload INTEGER)"
        )
        # create table to append per second data
        db.execute(
            "CREATE TABLE seconds (id INTEGER PRIMARY KEY, time REAL, particles INTEGER)"
        )

         # create table to append trajectory data per particle
        db.execute(
            "CREATE TABLE trajectories (id INTEGER PRIMARY KEY, x_init REAL, y_init REAL, area REAL, x_final REAL, y_final REAL, distance, REAL, moving_time REAL, speed REAL, first_frame REAL, last_frame REAL)"
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
        img = cv2.threshold(img, self.bin_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        # TODO: make threshold value configurable
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # TODO: make threshold value configurable
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        return img

    def write_to_sqlite(self, db_file, image_path, particles, pixel_length):

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
                "INSERT INTO particles (image, time, x, y, width, height, area, eccentricity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    image_path.as_posix(),
                    float(image_path.stem),
                    particle.centroid[1],
                    particle.centroid[0],
                    (particle.bbox[3] - particle.bbox[1])*pixel_length,
                    (particle.bbox[2] - particle.bbox[0])*pixel_length,
                    particle.area * (pixel_length**2),
                    particle.eccentricity,
                ),
            )

        # close database
        db.commit()
        db.close()

    def analyze_image(self, db_file, image_path, img_bg, pixel_length):
        particles = []

        # load image
        img = self.load_image_gray(image_path, crop=True)

        # subtract background
        img = cv2.subtract(img_bg, img)


        # stretch greyscale between 0 and 255
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # threshold image to isolate particles
        img = self.threshold_image(img)

        # moving the greyscale inversion to end because it was resulting in the whole image being white when testing
        img = cv2.bitwise_not(img)

        # find consecutive areas of white pixels
        particles = skimage.measure.label(img, background=0)
        particles = skimage.measure.regionprops(particles)

        # write to sqlite database
        self.write_to_sqlite(db_file, image_path, particles, pixel_length)

        df_particles = pd.DataFrame(
            [
                {
                    "time": float(image_path.stem),
                    "x": particle.centroid[1],
                    "y": particle.centroid[0],
                    "width": (particle.bbox[3] - particle.bbox[1])*pixel_length,
                    "height": (particle.bbox[2] - particle.bbox[0])*pixel_length,
                    "bbox": particle.bbox,
                    "area": particle.area * (pixel_length**2),
                    "eccentricity": particle.eccentricity,
                }
                for particle in particles
            ]
        )
        if not(df_particles.empty):
            df_particles.sort_values("area", ascending=False, inplace=True)

        return img, df_particles

    #TODO finish this work
    def calc_pixel_size(self, img_cal):

        # invert grayscale and stretch greyscale between 0 and 255
        img_cal = cv2.bitwise_not(img_cal)
        img_cal = cv2.normalize(img_cal, None, 0, 255, cv2.NORM_MINMAX)

        # find consecutive areas of white pixels
        dots = skimage.measure.label(img_cal, background=0)
        dots = skimage.measure.regionprops(dots)

        df_cal = pd.DataFrame(
            [
                {"x": dot.centroid[1], 
                 "y": dot.centroid[0]
                 }
                 for dot in dots
            ]
        )

        #print(df_cal)


        return None

    def run(self):
        #TODO: write code to obtain pixel to mm size
        pixel_length = self.calc_pixel_size(self.img_cal)

        #TODO speak to tobias about likelihood of this changing if elevation of lighttable camera and table won't change
        pixel_length = (15/45)

        frame_count = 0
        frame_count_total = 0
        tic = time.perf_counter()

        images = sorted(self.images)

        # load background image
        img_bg = self.load_image_gray(self.file_background, crop=True)

        img_time_start = float(images[0].stem)
        img_time_before = img_time_start
        img_prev = None
        df_part_prev = None

        # loop through images
        for image_path in images:
            img_time = float(image_path.stem)
            img, df_part_now = self.analyze_image(self.db_file, image_path, img_bg, pixel_length)

            frame_count += 1
            frame_count_total += 1

            # for every second of the experiment, print the fps and time remaining
            if np.floor(img_time) != np.floor(img_time_before):
                toc = time.perf_counter()
                fps = frame_count / (toc - tic)
                logger.info(
                    f"{img_time}, fps: {fps:.2f}, total frames: {frame_count_total}/{len(self.images)}, time left: {((len(self.images) - frame_count_total) / fps) / 60:.2f} min"
                )
                tic = time.perf_counter()
                frame_count = 0

            img_time_before = img_time

            #displays overlay of particle and its trakcs if "debug" is True, else not needed
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
                    p0 = np.float32(df_part_prev[["x", "y"]].values)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        img_prev, img, p0, None, **lk_params
                    )
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(
                        img, img_prev, p1, None, **lk_params
                    )
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 1

                    new_tracks = []
                    for tr, (x, y), good_flag in zip(p0, p1, good):
                        tr2 = (tr[0], tr[1])
                        if not good_flag:
                            continue

                        new_tracks.append(((tr2), (x, y)))

                    img_overlay = img.copy()
                    # overlay the previous image
                    img_overlay = cv2.addWeighted(img_overlay, 0.7, img_prev, 0.3, 0)
                    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_GRAY2BGR)

                    # draw the flow vectors
                    cv2.polylines(
                        img_overlay,
                        [np.int32(tr) for tr in new_tracks],
                        isClosed=False,
                        color=(0, 255, 0),
                    )

                    # display the image
                    cv2.imshow("img_overlay", img_overlay)

                    key = cv2.waitKey(200)

                if (img_time - img_time_start) > int(
                    self.c["debugging"]["stop_time_sec"]
                ):
                    cv2.waitKey()
                    break

                key = cv2.waitKey(200)
                if key == 27:  # if ESC is pressed, exit loop
                    cv2.destroyAllWindows()

                img_prev = img
                df_part_prev = df_part_now
