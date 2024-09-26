#Narrow Filter
# limits the possible combinations of particles before linking them together

# numerical processing
import numpy as np
import pandas as pd
import uuid
from scipy.optimize import linear_sum_assignment as lsa

# image importing
from pathlib import Path

# databse
import sqlite3

# for tracking progress
import time
import logging

# for debugging
import cv2

logger = logging.getLogger(__name__)

class NarrowFilter:      
    def __init__(self, c):
        self.images = list(Path(c["images"]["path"]).rglob("*.tif"))

        self.debug = c["debugging"]["debug"] == True

        # defining the config file
        self.c = c

        # writing the path to the sqlite database
        self.db_file = Path(
            c["output"]["path"],
            f'{c["config"]["run_name"]}{c["output"]["file_db_append"]}',
        )

        # connecting to sqlite database to create table for new data
        db = sqlite3.connect(self.db_file)
        
        #clear database
        db.execute("DROP TABLE IF EXISTS narrow_filter")

        # create table to append trajectory data per particle
        db.execute(
            "CREATE TABLE narrow_filter (id INTEGER PRIMARY KEY, x_init REAL, y_init REAL, area REAL, x_final REAL, y_final REAL, first_frame REAL, last_frame REAL)"
        )

        #close database
        db.commit()
        db.close()

        # setting the linking weights from the configuration file
        self.distance_weight = c["particle_linker"]["distance_weight"]
        self.area_weight = c["particle_linker"]["area_weight"]

        # setting the cost parameters
        self.max_cost = c["cost_parameters"]["max_cost"]
        self.max_frames = c["cost_parameters"]["max_frames"]

    def extract_particles(self, db_file, image_frame):
        # connecting to the database
        db = sqlite3.connect(db_file)

        cur = db.cursor()
        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: XX_filter -> id, x_init, y_init, area, x_final, y_final, first_frame, last_frame

        # extract x and y positions of particles
        cur.execute(
            f"SELECT x,y,area FROM particles WHERE (image = '{image_frame.as_posix()}')"
        )

        # saving list of lists of data extracted
        particle_properties = cur.fetchall()

        # converting lists into a numpy array
        particle_properties = np.array(particle_properties)

        return particle_properties

    def linking_particles(
        self, recent_df, particle_properties, image_frame):

        # finding number of new particles
        new_particle_count = len(particle_properties)

        #if there are no new particles update index counter of all the particles
        if new_particle_count == 0:
            recent_df['count'] += 1

        else:            
            #sort current particles by y position
            particle_properties = particle_properties[particle_properties[:,1].argsort()[::-1]]

            #sort recent_df by y position
            recent_df.sort_values(by="y_final", inplace=True, ascending=True)

            #empty list for storing the index of particles that left the lighttable
            particles_out_of_frame = []

            #iterate through recent_df and remove particles that have no particles beyond them (i.e. they are furthest up the screen)
            for index, current_row in recent_df.iterrows():
                
                #list of particles beyond the particle position stored in the dataframe
                current_possible_particles = particle_properties[particle_properties[:,1] < current_row["y_final"]]

                current_possible_particles = np.reshape(current_possible_particles, (-1,3))

                #if no particles are beyond the dataframe particle store it for trajectory tracking later
                if current_possible_particles.size == 0 or current_row["y_final"] < 200:
                    particles_out_of_frame.append(index)
                    continue
                
                #filter out possible particles by area so that they are more than half or less than double the area of the current particle
                current_possible_particles = current_possible_particles[np.logical_and(
                    current_row["area"] * 4 > current_possible_particles[:,2], 
                    current_row["area"] / 3 < current_possible_particles[:,2])]

                #filter out possible particles by x position so that they have not translated too far laterally
                #TODO: make this configurable and determine if value needs to scale by particle size or not
                current_possible_particles = current_possible_particles[np.logical_and(
                current_row["x_final"] + 4*current_row["area"] > current_possible_particles[:,0],
                current_row["x_final"] - 4*current_row["area"] < current_possible_particles[:,0])]

                #if there are no particles after filtering ignore the particle and continue, it may be the case that due to rotation it was no captured in the image
                #eventually the particle will hit the threshold for missing frames and be removed if necessary
                if current_possible_particles.size == 0:
                    recent_df.loc[index, "count"] += 1
                    continue

                #distance calculations
                distances = np.sqrt((current_row["x_final"] - current_possible_particles[:,0])**2 + 
                                    (current_row["y_final"] - current_possible_particles[:,1])**2)
                
                #area calculations
                area_change = np.abs(current_possible_particles[:,2] - current_row["area"]) / current_row["area"]

                #cost matrix
                cost_matrix = ((distances * self.distance_weight) + (area_change * self.area_weight))

                #if the cost matrix is below certain threshold (i.e. all particles are gone) then update counter for recent_df particle
                if cost_matrix.min() > self.max_cost:
                    recent_df.loc[index, "count"] += 1
                    continue
                
                #reshape cost matrix to append it
                cost_matrix = np.reshape(cost_matrix, (len(cost_matrix), 1))

                #append the cost_matrix to the list of possible particles
                current_possible_particles = np.append(current_possible_particles, cost_matrix, axis=1)

                #sort cost matrix for lowest cost to link
                current_possible_particles = current_possible_particles[current_possible_particles[:,3].argsort()]

                #update recent df with new values
                recent_df.loc[index, "x_final"] = int(current_possible_particles[0,0])
                recent_df.loc[index, "y_final"] = int(current_possible_particles[0,1])
                recent_df.loc[index, "area"] = current_possible_particles[0,2]
                recent_df.loc[index, "last_frame"] = image_frame
                recent_df.loc[index, "count"] = 1
 
                #remove the particle from the particle_properties array
                index_to_delete = np.where(np.all(particle_properties == current_possible_particles[0,:3], axis=1))[0]
                particle_properties = np.delete(particle_properties, index_to_delete, 0)

            #track the particles that left the image frame
            self.track_particles(self.db_file, recent_df.iloc[particles_out_of_frame])

            #droppping the particles from the recent df
            recent_df = recent_df.drop(particles_out_of_frame, axis=0)

            # reseting the index of the recent df
            recent_df = recent_df.reset_index(drop=True)

            #check if there are any leftover measured particles to track
            if particle_properties.size > 0:
                
                #create empty dataframe to store particle values so its easier to append to recent_df
                current_particle = pd.DataFrame(columns=[
                    "UID",
                    "first_frame",
                    "x_init",
                    "y_init",
                    "area",
                    "last_frame",
                    "x_final",
                    "y_final",
                    "count"])
                
                #loop through the particles and store their values in recent_df
                for remaining_particle in particle_properties:
                    current_particle.loc[0] = [uuid.uuid4(),
                                               image_frame,
                                               int(remaining_particle[0]),
                                               int(remaining_particle[1]),
                                               remaining_particle[2],
                                               image_frame,
                                               int(remaining_particle[0]),
                                               int(remaining_particle[1]),
                                               1]

                    # appending the new paricle to the recent df
                    recent_df = pd.concat([recent_df, current_particle], ignore_index=True)

                    # clearing the data from the current_particle df
                    current_particle = current_particle.drop(0)

            # logical test to check if any of the counters reach the max count
            count_test = np.logical_not(recent_df["count"] <= self.max_frames)

            # if the row reaches the max counter drop the row
            if any(count_test):

                # obtaining positions where particles have reached the max count
                count_index = recent_df.index[count_test]

                # takes the particles that are no longer tracked and inserts their paths to the trajectories database
                self.track_particles(self.db_file, recent_df.iloc[count_index])

                # droppping the particles from the recent df
                recent_df = recent_df.drop(count_index, axis=0)

                # reseting the index of the recent df
                recent_df = recent_df.reset_index(drop=True)

        return recent_df

    def track_particles(self, db_file, to_track):

        # connecting to the database
        db = sqlite3.connect(db_file)

        if not (to_track.empty):
            for index, row in to_track.iterrows():
                db.execute(
                    "INSERT INTO narrow_filter (x_init, y_init, area, x_final, y_final, first_frame, last_frame) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["x_init"],
                     row["y_init"],
                     row["area"],
                     row["x_final"],
                     row["y_final"],
                     row["first_frame"],
                     row["last_frame"]
                     ))

        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: XX_filter, id, x_init, y_init, area, x_final, y_final, first_frame, last_frame

        # close database
        db.commit()
        db.close()

    def run(self):
        frame_count = 1
        current_frame = 1

        images = sorted(self.images)

        # create a pandas dataframe to store images taken in the last second
        recent_df = pd.DataFrame(
            columns=["UID",
                     "first_frame",
                     "x_init",
                     "y_init",
                     "area",
                     "last_frame",
                     "x_final",
                     "y_final",
                     "count"])

        # start timer
        tic = time.perf_counter()

        # initializing the time of the first image
        time_before = float(images[0].stem)

        # total number of images in the directory
        frame_count_total = len(images)

        #create previous df to use for debugging
        prev_df = None

        #save the dimensions of the images for use in debugging
        img_row, img_col, img_dim = cv2.imread(images[0].as_posix()).shape[0:3]

        #need pixel length for accurate plotting
        pixel_length = 15 / 45

        print("starting to link the particles together")

        for image_path in images:
            particle_properties = self.extract_particles(self.db_file, image_path)

            # saving image time so it doesn't need to be done each time
            img_time = float(image_path.stem)

            # adding data to the data frame if its empty
            if recent_df.empty:
                for pp in range(len(particle_properties)):
                    recent_df.loc[pp] = [uuid.uuid4(),
                                         img_time,
                                         int(particle_properties[pp][0]),
                                         int(particle_properties[pp][1]),
                                         particle_properties[pp][2],
                                         img_time,
                                         int(particle_properties[pp][0]),
                                         int(particle_properties[pp][1]),
                                         1]

                print(f"{current_frame} was empty")

            else:
                recent_df = self.linking_particles(recent_df, particle_properties, img_time)
            current_frame += 1
            frame_count += 1

            # for every second of the experiment, print the fps and time remaining
            if np.floor(img_time) != np.floor(time_before):
                toc = time.perf_counter()
                fps = frame_count / (toc - tic)
                logger.info(
                    f"{img_time}, fps: {fps:.2f}, total frames: {current_frame}/{frame_count_total}, time left: {((frame_count_total - current_frame) / fps) / 60:.2f} min"
                )
                tic = time.perf_counter()
                frame_count = 0


            # displays overlay of particle if "debug" is True, else not needed
            if self.debug:

                if prev_df is not None:
                    
                    # print(prev_df)
                    # print(recent_df)

                    #create a black image the same size as the original image
                    blank_image = np.zeros((img_row, img_col, img_dim))

                    #plot the particles that weren't found in the recent dataframe on the blank image as red particles
                    lost_particles = prev_df[~prev_df['UID'].isin(recent_df['UID'])]

                    for index, particle in lost_particles.iterrows():
                        cv2.circle(blank_image, (particle['x_final'], particle['y_final']), int((np.sqrt(particle['area']/np.pi))/(pixel_length**2)), (0,0,255), -1)

                    #plot the new particles that weren't in the previous dataframe on the blank image as green partilces
                    new_particles = recent_df[~recent_df['UID'].isin(prev_df['UID'])]

                    for index, particle in new_particles.iterrows():
                        cv2.circle(blank_image, (particle['x_final'], particle['y_final']), int((np.sqrt(particle['area']/np.pi))/(pixel_length**2)), (0,255, 0), -1)

                    #plot the connected particles
                    linked_partilces = recent_df[recent_df['UID'].isin(prev_df['UID'])]

                    for index, particle in linked_partilces.iterrows():
                        if particle['count'] > 1:
                            cv2.circle(blank_image, (particle['x_final'], particle['y_final']), int((np.sqrt(particle['area']/np.pi))/(pixel_length**2)), (0,0,255), -1)
                        else:
                            cv2.circle(blank_image, (particle['x_final'], particle['y_final']), int((np.sqrt(particle['area']/np.pi))/(pixel_length**2)), (255, 0, 0), -1)
                            UID = particle['UID']
                            # print("This is the linked particle")
                            # print(prev_df[(prev_df['UID'] == UID)])
                    


                    # display the image
                    cv2.imshow("tracked particles", blank_image)

                    key = cv2.waitKey(1000)
                    cv2.waitKey(0)

                    # input("Press Enter to see next image")

                if (frame_count) > int(
                    self.c["debugging"]["images_to_view"]
                ):
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    self.debug = False

            time_before = img_time
            prev_df = recent_df

        # tracking any particles left in the dataframe after finished looking through all the images
        self.track_particles(self.db_file, recent_df)