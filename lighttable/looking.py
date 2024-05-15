import cv2
import numpy as np

def noise_filter(img, pixels=3):
    # filter out noise
    img = cv2.GaussianBlur(img, (pixels, pixels), 0)
    img = cv2.medianBlur(img, pixels)
    return img

def sharpen_image( img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img

def crop_image(img):    
    crop_left = 15
    crop_right = 15
    crop_top = 15
    crop_bottom = 15
    img = img[
                crop_top : img.shape[0] - crop_bottom,
                crop_left : img.shape[1] - crop_right,
            ]
    return img

def threshold_image(img, a, b):
    # threshold image to isolate particles
    # TODO: make threshold value configurable
    img = cv2.threshold(img, a, b, cv2.THRESH_BINARY_INV)[1]
    # TODO: make threshold value configurable
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # TODO: make threshold value configurable
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    return img
    
    
img_bg = cv2.imread("background.tif")
img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
img_bg = noise_filter(img_bg)
img_bg = sharpen_image(img_bg)
img_bg = crop_image(img_bg)

cv2.imshow("background_image", img_bg)
cv2.waitKey(0)


# img = cv2.imread("t00-t20/3583756189.772821.tif")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = noise_filter(img)
# img = sharpen_image(img)
# img = crop_image(img)

# cv2.imshow("raw_image", img)
# cv2.waitKey(0)

# sub1 = cv2.subtract(img_bg, img)

# #sub1 = cv2.bitwise_not(sub1)
# sub1 = cv2.normalize(sub1, None, 0, 255, cv2.NORM_MINMAX)

# #sub1 = threshold_image(sub1)

# cv2.imshow("subtract img from background", sub1)
# cv2.waitKey(0)

# sub2 = img - img_bg

# sub2 = cv2.bitwise_not(sub2)
# sub2 = cv2.normalize(sub2, None, 0, 255, cv2.NORM_MINMAX)

# #sub2 = threshold_image(sub2)

# cv2.imshow("subtract background from img", sub2)
# cv2.waitKey(0)

# inv_img = cv2.bitwise_not(img)

# cv2.imshow("inverted image", inv_img)
# cv2.waitKey(0)

# threshold_img = threshold_image(img, 50, 255)

# cv2.imshow("threshold_raw", threshold_img)
# cv2.waitKey(0)



img = cv2.imread("t00-t20/3583756189.772821.tif")
cv2.imshow("step 1", img)
cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("step 2", img)
cv2.waitKey(0)

img = noise_filter(img)
cv2.imshow("step 3", img)
cv2.waitKey(0)

img = sharpen_image(img)
cv2.imshow("step 4", img)
cv2.waitKey(0)

img = crop_image(img)
cv2.imshow("step 5", img)
cv2.waitKey(0)

img = cv2.subtract(img_bg, img)
cv2.imshow("step 6", img)
cv2.waitKey(0)





img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("step 8", img)
cv2.waitKey(0)

# threshold image to isolate particles
img1 = threshold_image(img, 160, 255)
cv2.imshow("160", img1)
cv2.waitKey(0)


# threshold image to isolate particles
img2 = threshold_image(img, 180, 255)
cv2.imshow("180", img2)
cv2.waitKey(0)

# threshold image to isolate particles
img3 = threshold_image(img, 200, 255)
cv2.imshow("200", img3)
cv2.waitKey(0)

# threshold image to isolate particles
img4 = threshold_image(img, 220, 255)
cv2.imshow("220", img4)
cv2.waitKey(0)


# invert grayscale and stretch greyscale between 0 and 255
img = cv2.bitwise_not(img)
cv2.imshow("step 10", img)
cv2.waitKey(0)