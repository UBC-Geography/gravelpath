# Description: This script is used to process images of a light table and identify
# stones in subsequent images. It is used to measure the size of stones and ultimately
# to count the number of stones in each image for sediment transport studies.
# 2024-03-10: Tobias Mueller, initial version

import logging
from pathlib import Path
from datetime import datetime
import time


# image processing
import cv2
import numpy as np
import skimage

# plotting
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class ImageLooper:
    def __init__(self, c):
        self.images = list(Path(c["images"]["path"]).rglob("*.tif"))

        self.file_background = Path(c["images"]["file_background"])
        self.file_calibration = Path(c["images"]["file_calibration"])
        self.debug = c["debugging"]["debug"]
        self.c = c

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

    # threshold image to isolate stones<ctrl63>
    def threshold_image(self, img):
        # threshold image to isolate stones
        img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)[1]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        # img = cv2.dilate(img, np.ones((3, 3), np.uint8))
        # img = cv2.erode(img, np.ones((3, 3), np.uint8))
        return img

    def run(self):
        # load background and calibration images
        img_bg = self.load_image_gray(self.file_background, crop=False)
        img_cal = self.load_image_gray(self.file_calibration, crop=False)

        # sort images by name
        self.images.sort()

        img_prev = None
        stones_prev = None

        # loop through images
        for image_path in self.images:
            tic = time.perf_counter()
            logger.info(f"Running image: {image_path}")

            # load image
            img = self.load_image_gray(image_path, crop=True)

            # subtract background
            img = cv2.subtract(img_bg, img)

            # invert grayscale and stretch greyscale between 0 and 255
            img = cv2.bitwise_not(img)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

            if self.debug:
                cv2.imshow("Subtracted", img)
                cv2.moveWindow("Subtracted", 0, 0)

                # show histogram of color distribution
                plt.hist(img.ravel(), 256, [0, 256])
                # plt.show()

            # threshold image to isolate stones
            img = self.threshold_image(img)

            if self.debug:
                cv2.imshow("Thresholded", img)
                cv2.moveWindow("Thresholded", 0, 0)

            # find consecutive areas of white pixels
            stones = skimage.measure.label(img, background=0)
            stones = skimage.measure.regionprops(stones)

            # show all contours on the image
            if self.debug:
                # outline coloured by stone area
                area_min = min([stone.area for stone in stones])
                area_max = max([stone.area for stone in stones])
                cmap = plt.cm.get_cmap("viridis")

                for stone in stones:
                    area = stone.area
                    if area <= 4:
                        continue

                    cv2.rectangle(
                        img,
                        (int(stone.bbox[1]), int(stone.bbox[0])),
                        (int(stone.bbox[3]), int(stone.bbox[2])),
                        (int(cmap((area - area_min) / (area_max - area_min))[0] * 255)),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"{area:.0f}",
                        (int(stone.bbox[1]), int(stone.bbox[0])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                cv2.imshow("Contours", img)
                cv2.moveWindow("Contours", 0, 0)

            toc = time.perf_counter()
            logger.info(
                f"Finished running image: {image_path}, took {toc-tic:0.4f} seconds"
            )
            img_prev = img

            key = cv2.waitKey(200)
            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
