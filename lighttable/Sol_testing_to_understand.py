import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sqlite3
import skimage
import pandas as pd
import uuid
from scipy.optimize import linear_sum_assignment as lsa 

image_path = "/Users/sol/Desktop/Kevin_Experiment/short_boat_stack/vid-120fps-07-05-22.1_Cam_11363_Cine1000000.tif"
image_path_2 = "/Users/sol/Desktop/Kevin_Experiment/short_boat_stack/vid-120fps-07-05-22.1_Cam_11363_Cine1000001.tif"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread(image_path_2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def sharpen_image(img):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img

img1 = sharpen_image(img)
# cv2.imshow('sharpened image', img1)
# cv2.waitKey(0)

img3 = sharpen_image(img2)
# cv2.imshow('sharpened image', img1)
# cv2.waitKey(0)

def db_create(db_path):
    db = sqlite3.connect(db_path)

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

db_create("/Users/Sol/Desktop/test_output.sqlite")


def draw_flow(img, flow, step=4):

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

flow = cv2.calcOpticalFlowFarneback(
                        img1, img3, None, 0.5, 5, 12, 3, 5, 1.2, 0
                    )

cv2.imshow('flow', flow[:,:,0])
cv2.waitKey(0)

cv2.imshow('flow 2', flow[:,:,1])
cv2.waitKey(0)

for ii in range(5):
    print(ii)

draw_flow(img3, flow)
y, x = (
            np.mgrid[4 / 2 : 720 : 4, 4 / 2 : 960 : 4]
            .reshape(2, -1)
            .astype(int)
        )
fx, fy = flow[y, x].T
flow_mag = np.sqrt(fx**2 + fy**2)

fx, fy = fx[flow_mag > 1], fy[flow_mag > 1]
x, y = x[flow_mag > 1], y[flow_mag > 1]

#create and then draw lines displaying the dense optical flow's calculated velocity
lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
lines = np.int32(lines + 0.5)
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
vis = cv2.polylines(vis, lines, 0, (0, 0, 255))

cv2.imshow('velocity lines', vis)
cv2.waitKey(0)

x.shape
y.shape

cv2.imshow('flow_mag', flow_mag)
cv2.waitKey(0)

flow_mag = flow_mag.reshape(180,240)

lines.shape
vis.shape

empty_array = np.zeros((720,960))
lines[0,0,0]
x
y

for ii,nn in enumerate(flow_mag[flow_mag>1]):
    a = x[ii]
    b = y[ii]
    empty_array[b,a] = 255

particles = skimage.measure.label(particle_array, background=0)
particles = skimage.measure.regionprops(particles)


df_particles = pd.DataFrame(
            [
                {
                    "x": particle.centroid[1],
                    "y": particle.centroid[0],
                    "width": particle.bbox[3] - particle.bbox[1],
                    "height": particle.bbox[2] - particle.bbox[0],
                    "bbox": particle.bbox,
                    "area": particle.area,
                }
                for particle in particles
            ]
        )
#df_particles.sort_values("area", ascending=False, inplace=True)

df_particles


# TODO: make threshold value configurable
particle_array = cv2.morphologyEx(empty_array, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
cv2.imshow('morph close 4', particle_array)
cv2.waitKey(0)

particle_array = cv2.morphologyEx(empty_array, cv2.MORPH_TOPHAT, np.ones((4, 4), np.uint8))
cv2.imshow('morph tophat 4', particle_array)
cv2.waitKey(0)

particle_array = cv2.morphologyEx(empty_array, cv2.MORPH_BLACKHAT, np.ones((4, 4), np.uint8))
cv2.imshow('morph BLACKHAT 4', particle_array)
cv2.waitKey(0)

particle_array = cv2.morphologyEx(empty_array, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
cv2.imshow('morph open', particle_array)
cv2.waitKey(0)

particle_array = cv2.dilate(empty_array, np.ones((4, 4)))
cv2.imshow('morph dilate 3', particle_array)
cv2.waitKey(0)

particle_array = cv2.dilate(particle_array, np.ones((2, 2)))
cv2.imshow('morph dilate 4', particle_array)
cv2.waitKey(0)

particle_array = cv2.erode(particle_array, np.ones((3, 3)))
cv2.imshow('morph dilate 3 erode 2', particle_array)
cv2.waitKey(0)

cv2.destroyAllWindows()


plt.imshow(particle_array, cmap='gray')
for ii in range(len(df_particles)):
    minr, minc, maxr, maxc = df_particles.loc[ii, 'bbox']
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)
plt.savefig('test boundbox')

plt.imshow(img3, cmap='gray')
for ii in range(len(df_particles)):
    minr, minc, maxr, maxc = df_particles.loc[ii, 'bbox']
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)
plt.savefig('test boxes on image')

plt.imshow(vis, cmap='gray')
plt.savefig('initial flow')

df_particles.drop[df_particles["area"]<100][0]

len(df_particles["area"])

df_particles['area'][df_particles["area"]>100, 'area']

df_particles['area']

np.where(df_particles['area']<100)

df_particles.drop([5])


con = sqlite3.connect('output/manual_mask_test_output.sqlite')

cur = con.cursor()

# cur.execute(f"SELECT x, y FROM particles WHERE (time = '{1/120}')")
cur.execute(f"SELECT x_init, y_init, area, x_final, y_final, moving_time, speed, first_frame, last_frame FROM trajectories")

coords = cur.fetchall()

print(coords)

np.savetxt("trajectories.csv", coords, delimiter=',')

recent_particles = pd.DataFrame(columns = ['UID', 'first_frame', 'x_init', 'y_init', 'last_frame', 'x_pos', 'y_pos', 'count'])

# for ii in range(len(coords)):
recent_particles.loc[0] = [uuid.uuid4(), 0, coords[ii][0], coords[ii][1], 0, coords[ii][0], coords[ii][1], 1]
recent_particles.loc[0] = [uuid.uuid4(), 0, coords[ii][0], coords[ii][1], 0, coords[ii][0], coords[ii][1], 1]
    #uuid.uuid4()x_coord_this_image[ii]

recent_particles

# plt.figure(figsize=(20,10))
# for ii in range(10):
#     plt.scatter(x_coords[ii], y_coords[ii], label = str(ii))
# plt.ylim(720,0)
# plt.legend()
# plt.savefig("coordinates")



# plt.figure(figsize=(20,10))
# plt.bar(range(19), particles_per_image)
# plt.title("Distribution of instantaneous counts of moving partilces")
# plt.ylabel("Count")
# plt.xlabel("Number of Moving Particles")
# plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
# plt.savefig("movement_distribution")

coins = skimage.data.binary_blobs()
particles = skimage.measure.label(coins, background=200)
particles = skimage.measure.regionprops(particles)
print(particles)
len(particles)
df_particles = pd.DataFrame(
            [
                {
                    "x": particle.centroid[1],
                    "y": particle.centroid[0],
                    "width": particle.bbox[3] - particle.bbox[1],
                    "height": particle.bbox[2] - particle.bbox[0],
                    "bbox": particle.bbox,
                    "area": particle.area,
                }
                for particle in particles
            ])

plt.imshow(coins, cmap='gray')
for ii in range(len(df_particles)):
    minr, minc, maxr, maxc = df_particles.loc[ii, 'bbox']
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    plt.plot(bx, by, '-b', linewidth=2.5)
plt.savefig('test boundbox')

for particle in particles: 
     particle.area

sum(particles[0].area <100)

count = 0
count =+1
count += 1
count

uuid.uuid4()

a = pd.DataFrame(columns = ["q1", "q2", "q3"])
a.loc[0] = [2,3,4]
a.loc[1] = [5,6,7]
a.loc[2] = [8,9,10]
a["q1"] == 2
a.index[np.logical_and(a["q1"] > 3, a["q3"] < 9)]
a.loc1
a["q1"] > 3

a.loc[a["q1"] == 2]["q1"] = 4
np.indices()
b = a.index[a["q1"] == 2]
a.loc[b, ["q1", "q2"]] = [4,12]
a["q1"]

b
len(a)
a.index[a["q1"] > 2]

any([False, False, False])


a.loc[[1,2], "q1"] += 1

np.logical_not(a["q1"] == 2)

a
a.drop(0, axis = 0)

a = a.drop(0, axis = 0)

img1 = cv2.imread("Mask1.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

particles = skimage.measure.label(img1, background=0)
particles = skimage.measure.regionprops(particles)
particles_table2 = skimage.measure.regionprops_table(particles, properties = ['Eccentricity'])

pd.DataFrame(particles_table2)
particles[0].eccentricity
particles[0].centroid

# Properties avaialble to be measured by Matlab are
# Area, BoundingBox, Centroid, Circularity, ConvexArea, ConvexImage
# Eccentricity, EquivDiameter, EulerNumber, Extent, Extrema, 


# compare to previous image to connect the particle
# minimise euclidian distance
# minimise change in eccentricity
# minimise change in area
# minimise major axis length???

# for a single particle loop through particles in previous image partilces 

distance = pd.DataFrame(columns = ["test1", "test2"])

distance['test1'] = np.sqrt((2-a['q1'])**2 + (5-a['q3'])**2)

distance.insert(2, "test4", np.sqrt((10-a['q1'])**2 + (9-a['q3'])**2))

distance = pd.DataFrame()
distance.insert(0, "hello", np.sqrt((10-a['q1'])**2 + (9-a['q3'])**2))


a = pd.DataFrame(columns = ["q1", "q2", "q3", "q4"])
a.loc[0] = [2,3,4,7]
a.loc[1] = [5,6,7, 2]
a.loc[2] = [8,9,10, 1]

b = pd.DataFrame(columns = ["q1", "q2", "q3", "q4"])
b.loc[0] = [22,12,30, 10]
b.loc[1] = [14,30,16, 0]
b.loc[2] = [21,40,19, 30]

c = pd.DataFrame(a+b)

c

row_ind, col_ind = lsa(c)
row_ind
col_ind

c.loc[0]["q1"]

lsa(c)

c.loc[1, 'q1'] = 2
c



recent_df = pd.DataFrame(columns = ['UID', 'first_frame', 'x_init', 'y_init', 'area', 'eccentricity',
                                                 'last_frame', 'x_final', 'y_final', 'count'])

len(recent_df.columns)

recent_df.columns[1]

for ii, col in enumerate(recent_df.columns):
    print(col)

g = np.array([1,2,3,4])

pd.concat([c,g], ignore_index=True)

c = c.drop(0)

for ii in []:
    print("hello")

c.reset_index()

img = cv2.imread('Mask0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

particles = skimage.measure.label(img, background=0)
particles = skimage.measure.regionprops(particles)

for particle in particles:
    print(particle.help)

particles.__class__

dir(particle)
particle.axis_major_length
particle.axis_minor_length

#TODO

def noise_filter(img, pixels=3):
    # filter out noise
    img = cv2.GaussianBlur(img, (pixels, pixels), 0)
    img = cv2.medianBlur(img, pixels)
    return img

def sharpen_image(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img


calibration_image = cv2.imread("calibration_crop.tif")
calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
calibration_image = noise_filter(calibration_image)
calibration_image = sharpen_image(calibration_image)

#cv2.imwrite("imported_calibration_image.tif", calibration_image)

#TODO finish this work
def calc_pixel_size(img_cal):

    # invert grayscale and stretch greyscale between 0 and 255
    img_cal = cv2.bitwise_not(img_cal)
    img_cal = cv2.normalize(img_cal, None, 0, 255, cv2.NORM_MINMAX)
    
    print(img_cal)

    img_cal = cv2.subtract(img_cal, 150)

    print(img_cal)

    # find consecutive areas of white pixels
    dots = skimage.measure.label(img_cal, background = 0)
    dots = skimage.measure.regionprops(dots)

    print(dir(dots[0]))

    df_cal = pd.DataFrame(
        [
            {"x": dot.centroid[1], 
             "y": dot.centroid[0], 
             "area": dot.area,
             "axis_length": dot.axis_major_length,
             "eccenricity": dot.eccentricity

                }
                for dot in dots
        ]
    )

    print(df_cal)

    return df_cal, dots

df_cal, dots = calc_pixel_size(calibration_image)

df_cal.max(axis=0)
df_cal[df_cal['area']==154]


fig, ax = plt.subplots()
ax.imshow(calibration_image)

for dot in dots:
    y0, x0 = dot.centroid
    orientation = dot.orientation

    #ax.plot(x0, y0, '.g', markersize=1)

    minr, minc, maxr, maxc = dot.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', color='red', linewidth=1)

y0,x0= dots[28851].centroid

ax.plot(x0,y0, color='g')

minr, minc, maxr, maxc = dots[28851].bbox
bx = (minc, maxc, maxc, minc, minc)
by = (minr, minr, maxr, maxr, minr)
ax.plot(bx, by, '-b', color='red', linewidth=0.5)

plt.savefig("region_cal_image.png")

df_cal.max(axis=0)
df_cal.min(axis=0)
big_enough = df_cal.index[df_cal['area'] > 60]

df_cal.loc[big_enough].max(axis=0)

sum(df_cal.loc[big_enough]['area'] <50)

len(big_enough)

sum(df_cal['area'] > 30)

calibration_image - 200

sum(df_cal['x'] < 30)

df_cal = df_cal.sort_values(by='area', ascending=False)

df_cal = df_cal.reset_index(drop=True)

df_cal = df_cal.loc[0:1225]
df_cal.sort_values(by='x')


cal_dot_mat = np.zeros((35,35))

