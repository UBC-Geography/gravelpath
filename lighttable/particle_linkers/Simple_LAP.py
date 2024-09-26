#SimpleLAP
# Uses linear sum assignment to match particles minimizing overall cost

# numerical processing
import numpy as np
import pandas as pd
import uuid
from scipy.optimize import linear_sum_assignment as lsa

#image importing
from pathlib import Path

#databse
import sqlite3

#for tracking progress
import time
import logging

# for debugging
import cv2

logger = logging.getLogger(__name__)

class SimpleLAP:
    def __init__(self, c):
        
        self.images = list(Path(c["images"]["path"]).rglob("*.tif"))

        self.debug = c["debugging"]["debug"] == True
        
        #defining the config file
        self.c = c

        #writing the path to the sqlite database
        self.db_file = Path(
            c["output"]["path"],
            f'{c["config"]["run_name"]}{c["output"]["file_db_append"]}',
        )
        
        # connecting to sqlite database to create table for new data
        db = sqlite3.connect(self.db_file)
        
        #clear database
        db.execute("DROP TABLE IF EXISTS simple_lap")

        # create table to append trajectory data per particle
        db.execute(
            "CREATE TABLE simple_lap (id INTEGER PRIMARY KEY, x_init REAL, y_init REAL, area REAL, x_final REAL, y_final REAL, first_frame REAL, last_frame REAL)"
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
        #connecting to the database
        db = sqlite3.connect(db_file)

        cur = db.cursor()
        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: XX_filter -> id, x_init, y_init, area, x_final, y_final, first_frame, last_frame

        #extract x and y positions of particles
        cur.execute(f"SELECT x,y,area FROM particles WHERE (image = '{image_frame.as_posix()}')")

        #saving list of lists of data extracted
        particle_properties = cur.fetchall()

        #converting lists into a numpy array
        particle_properties = np.array(particle_properties)

        return particle_properties
    

    def linking_particles(self, recent_df, particle_properties, image_frame):

        #finding number of new and old partilces
        new_particle_count = len(particle_properties)
        old_particle_count = len(recent_df)

        #extracting data for vectorization from recent df
        x_final = recent_df['x_final'].values
        y_final = recent_df['y_final'].values
        areas = recent_df['area'].values

        # Initialize arrays to store the computed values for vectorization
        distances = np.zeros((old_particle_count, new_particle_count))        
        area_changes = np.zeros((old_particle_count, new_particle_count))


        for ii in range(new_particle_count):
            # Computing Euclidean distance
            distances[:,ii] = np.sqrt((particle_properties[ii, 0] - x_final) ** 2 + 
                                    (particle_properties[ii, 1] - y_final) ** 2)
            
            # Computing percentage change in area
            area_changes[:,ii] = np.abs(1 - (particle_properties[ii, 2] / areas))

        # Create DataFrames from the arrays
        distance_df = pd.DataFrame(distances, columns=[f"particle {ii+1}" for ii in range(new_particle_count)])
        area_change_df = pd.DataFrame(area_changes, columns=[f"particle {ii+1}" for ii in range(new_particle_count)])
        # eccentricity_change_df = pd.DataFrame(eccentricity_changes, columns=[f"particle {ii+1}" for ii in range(new_particle_count)])
        current_particle = pd.DataFrame(
            columns = ['UID', 
                       'first_frame', 
                       'x_init', 
                       'y_init', 
                       'area', 
                       'last_frame', 
                       'x_final', 
                       'y_final', 
                       'count'])

        #computing the cost matrix
        cost_matrix = pd.DataFrame((distance_df * self.distance_weight) + (area_change_df * self.area_weight)) 

        #solving the cost matrix for the optimal solution (row_ind corresponds to old particles, col_ind corresponds to new particles)
        row_ind, col_ind = lsa(cost_matrix)

        #obtaining a list of old particles that do not have a match
        lost_particles = list(range(old_particle_count))
        for ii in row_ind:
            lost_particles.remove(ii)

        #obtaining a list of new particles
        new_particles = list(range(new_particle_count))
        for ii in col_ind:
            new_particles.remove(ii)     

        #linking particles using the cost_matrix
        for ii in range(len(row_ind)):
            if cost_matrix.iloc[row_ind[ii],col_ind[ii]] < self.max_cost:

                #importing new values into the recent particle df
                recent_df.loc[row_ind[ii],'x_final'] = int(particle_properties[col_ind[ii]][0])
                recent_df.loc[row_ind[ii],'y_final'] = int(particle_properties[col_ind[ii]][1])
                recent_df.loc[row_ind[ii],'area'] = particle_properties[col_ind[ii]][2]
                recent_df.loc[row_ind[ii],'last_frame'] = image_frame
                recent_df.loc[row_ind[ii],'count'] = 1

            else:
                #TODO determine if we need to update with possible position
                recent_df.loc[row_ind[ii],'count'] += 1

                #extracting properties of the new particle          
                current_particle.loc[0] = [uuid.uuid4(), 
                                           image_frame, 
                                           int(particle_properties[col_ind[ii]][0]), 
                                           int(particle_properties[col_ind[ii]][1]), 
                                           particle_properties[col_ind[ii]][2], 
                                           image_frame, 
                                           int(particle_properties[col_ind[ii]][0]), 
                                           int(particle_properties[col_ind[ii]][1]), 
                                           1]
        
                #appending the new paricle to the recent df
                recent_df = pd.concat([recent_df, current_particle], ignore_index=True)

                #clearing the data from the current_particle df
                current_particle = current_particle.drop(0)

        #if matrix is not a square will update counter for the particles in the recent df that didn't match a new one
        for ii in lost_particles:
            #TODO update with new positions
            recent_df.loc[ii,'count'] += 1
 
        #if matrix is not a square will add the extra particles form the new image to recent df 
        for ii in new_particles:

            #extracting properties of the new particle        
            current_particle.loc[0] = [uuid.uuid4(),
                                       image_frame, 
                                       int(particle_properties[ii][0]), 
                                       int(particle_properties[ii][1]),
                                       particle_properties[ii][2], 
                                       image_frame, 
                                       int(particle_properties[ii][0]), 
                                       int(particle_properties[ii][1]), 
                                       1]
            
            #appending the new paricle to the recent df
            recent_df = pd.concat([recent_df, current_particle], ignore_index=True)

            #clearing the data from the current_particle df
            current_particle = current_particle.drop(0)

        #logical test to check if any of the counters reach the max count
        count_test = np.logical_not(recent_df["count"] <= self.max_frames)

        #if the row reaches the max counter drop the row
        if any(count_test):

            #obtaining positions where particles have reached the max count
            count_index = recent_df.index[count_test]

            #takes the particles that are no longer tracked and inserts their paths to the trajectories database 
            self.track_particles(self.db_file, recent_df.iloc[count_index])

            #droppping the particles from the recent df
            recent_df = recent_df.drop(count_index, axis = 0)

            #reseting the index of the recent df
            recent_df = recent_df.reset_index(drop=True)

        return recent_df, cost_matrix
    
    def track_particles(self, db_file, to_track):
        
        #connecting to the database
        db = sqlite3.connect(db_file)

        if not(to_track.empty):
            for index, row in to_track.iterrows():
                db.execute(
                    "INSERT INTO simple_lap (x_init, y_init, area, x_final, y_final, first_frame, last_frame) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (   row['x_init'],
                        row['y_init'],
                        row['area'],
                        row['x_final'],
                        row['y_final'],
                        row['first_frame'],
                        row['last_frame']
                        ))

        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: XX_Filter, id, x_init, y_init, area, x_final, y_final, first_frame, last_frame

        # close database
        db.commit()
        db.close()
    

    def run(self):
        frame_count = 1
        current_frame =1

        images = sorted(self.images)

        #create a pandas dataframe to store images taken in the last second
        recent_df = pd.DataFrame(
            columns = ['UID', 
                       'first_frame', 
                       'x_init', 
                       'y_init', 
                       'area', 
                       'last_frame', 
                       'x_final', 
                       'y_final', 
                       'count'])
        
        #start timer
        tic = time.perf_counter()

        #initializing the time of the first image
        time_before = float(images[0].stem)

        #total number of images in the directory
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
            #print(particle_properties)

            #saving image time so it doesn't need to be done each time
            img_time = float(image_path.stem)

            #adding data to the data frame if its empty
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
                recent_df, cost_matrix = self.linking_particles(recent_df, particle_properties, img_time)
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

        #tracking any particles left in the dataframe after finished looking through all the images
        self.track_particles(self.db_file, recent_df)
