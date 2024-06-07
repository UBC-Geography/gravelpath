## Lighttable method

# particle one at a time and predict movement

## Stochastic method

# numerical processing
import numpy as np
import pandas as pd
import uuid
from scipy.optimize import linear_sum_assignment as lsa
import sys

#image importing
from pathlib import Path

#databse
import sqlite3

#for plotting
import matplotlib.pyplot as plt

#for tracking progress
import time
import logging

logger = logging.getLogger(__name__)

class Particle_Loop:
    def __init__(self, c):
        
        self.images = list(Path(c["images"]["path"]).rglob("*.tif"))

        self.debug = c["debugging"]["debug"] == "True"
        
        #defining the config file
        self.c = c

        #writing the path to the sqlite database
        self.db_file = Path(
            c["output"]["path"],
            f'{c["config"]["run_name"]}{c["output"]["file_db_append"]}',
        )
        
        #TODO determine if the number of frames needs to initialized here
        #total number of frames needed to look over
        #self.total_frames = c["camera"]["frames"] - 1
        self.fps = c['camera']['framerate']

        #setting the linking weights from the configuration file
        self.distance_weight = c["linking_weight"]["distance"]
        self.area_weight = c["linking_weight"]["area"]
        self.eccentricity_weight = c["linking_weight"]["eccentricity"]
        
        #setting the cost parameters
        self.max_cost = c['cost_parameters']['max_cost']
        self.max_frames = c['cost_parameters']['max_frames']
 
        #setting values needed for kalman filter
        min_normal_float = sys.float_info.min               #smallest possible normal number
        search_radius = c['cost_parameters']['search_radius']
        process_noise = (1/3) * search_radius

        #initializing matrices for kalman filter
        self.evolution_matrix = np.array([[1, 0, 1, 0],     #4x4 matrix for propogating measurements
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])
        self.measurement_matrix = np.eye(4)                 #4x4 matrix for propogating measurements
        self.state_covariance = np.array([[min_normal_float, 0, 0, 0],   #4x4 matrix for state covariance
                                         [0, min_normal_float, 0, 0],
                                         [0, 0, min_normal_float, 0],
                                         [0, 0, 0, min_normal_float]])
        self.process_covariance = np.eye(4) * process_noise #4x4 matrix for covariance of processing
        #TODO fix this 
        #observation_noise = (1/10)*mean_particle_radius

        #self.observation_covariance = np.eye(4)*(observation_noise**2) #4x4 matrix for covariance of measurements

        

    def extract_particles(self, db_file, image_frame, particle_df):
        #connecting to the database
        db = sqlite3.connect(db_file)

        cur = db.cursor()
        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: trajectories, id, x_init, y_init, area, x_final, y_final, distance, moving_time, speed, first_frame, last_frame

        
        #extract x and y positions of particles
        cur.execute(f"SELECT x,y,area FROM particles WHERE (image = '{image_frame.as_posix()}')")

        #saving list of lists of data extracted
        particle_properties = cur.fetchall()

        #converting lists into a numpy array
        particle_properties = np.array(particle_properties)

        return particle_properties
    

    def linking_particles(self, recent_df, particle_properties, current_frame, image_frame):

        #finding number of new and old partilces
        new_particle_count = len(particle_properties)
        old_particle_count = len(recent_df)


        #extracting data for vectorization from recent df
        x_previous = recent_df['x_recent'].values
        y_previous = recent_df['y_recent'].values
        x_vel = recent_df['x_velocity'].values
        y_vel = recent_df['y_velocity'].values

        #setting up array to store predicted positions
        predicted_positions = np.zeros((old_particle_count, 2))

        #predicting position for each particle in recent_df
        for ii in range(old_particle_count):
            #creating state matrix for kalman filter
            state_matrix = np.array([[x_previous[ii]],
                                [y_previous[ii]], 
                                [x_vel[ii]], 
                                [y_vel[ii]]])

            #predicting positions using kalman filter
            predicted_state =  self.kalman_predict(state_matrix)

            #saving predicted positions to the array for calculations
            predicted_positions[ii,0:2] = predicted_state[0:2]
            recent_df.loc[ii,'x_recent'] = predicted_state[0]
            recent_df.loc[ii,'y_recent'] = predicted_state[1]

        #logical test to see if any of the predicted positiosn are outside of the image
        in_frame_test = np.logical_and(abs(recent_df['y_recent']) > self.c['cost_parameters']['prediction_error'], recent_df['y_recent'] < 0)
        
        #obtaining the positions of the partiles that are likely outside of the frame
        if any(in_frame_test):
            out_of_frame_index = recent_df.index[in_frame_test]

            #takes the particles that are out of frame and inserts their paths to the trajectories database 
            self.track_particles(self.db_file, recent_df.iloc[out_of_frame_index])
            
            #droppping the particles from the recent df
            recent_df = recent_df.drop(out_of_frame_index, axis = 0)

            #reseting the index of the recent df
            recent_df = recent_df.reset_index(drop=True)
    
        #updating the number of old particles after removing the ones that left the frame
        old_particle_count = len(recent_df)

        # Initialize arrays to store the computed values for vectorization
        distances = np.zeros((old_particle_count, new_particle_count))        
        area_changes = np.zeros((old_particle_count, new_particle_count))

        for ii in range(new_particle_count):

            # Computing Euclidean distance
            distances[:,ii] = np.sqrt((particle_properties[ii, 0] - recent_df['x_recent']) ** 2 + 
                                      (particle_properties[ii, 1] - recent_df['y_recent']) ** 2)
            
            # Computing percentage change in area
            area_changes[:,ii] = np.abs(1 - (particle_properties[ii, 2] / recent_df['area']))

        # Create DataFrames from the arrays
        distance_df = pd.DataFrame(distances, columns=[f"particle {ii+1}" for ii in range(new_particle_count)])
        area_change_df = pd.DataFrame(area_changes, columns=[f"particle {ii+1}" for ii in range(new_particle_count)])
        current_particle = pd.DataFrame(columns = ['UID', 'first_frame', 'x_init', 'y_init', 'area',
                                                'last_frame', 'x_recent', 'y_recent', 'count', 'x_velocity', 'y_velocity'])

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
                recent_df.loc[row_ind[ii],'x_velocity'] = int(particle_properties[col_ind[ii]][0]) - recent_df.loc[row_ind[ii],'x_recent'] 
                recent_df.loc[row_ind[ii],'y_velocity'] = int(particle_properties[col_ind[ii]][1]) - recent_df.loc[row_ind[ii],'y_recent'] 
                recent_df.loc[row_ind[ii],'x_recent'] = int(particle_properties[col_ind[ii]][0])
                recent_df.loc[row_ind[ii],'y_recent'] = int(particle_properties[col_ind[ii]][1])
                recent_df.loc[row_ind[ii],'area'] = particle_properties[col_ind[ii]][2]
                recent_df.loc[row_ind[ii],'last_frame'] = image_frame
                recent_df.loc[row_ind[ii],'count'] = 1

            # #if the particle was not linked to another particle and the predicted position is out of frame then we should send the particle 
            # #data to trajectories and remove it from the recent df as it will definitely be out of frame by the next image
            # elif recent_df.loc[row_ind[ii], 'y_recent'] < 0: 
            #     #sending the data to the sqlite database
            #     print(recent_df.iloc[ii])
            #     self.track_particles(self.db_file, recent_df.iloc[ii])

            #     #droppping the particle from the recent df
            #     recent_df = recent_df.drop(ii, axis = 0)

            else: 
                recent_df.loc[row_ind[ii],'count'] += 1

                #extracting properties of the new particle          
                current_particle.loc[0] = [uuid.uuid4(), image_frame, int(particle_properties[col_ind[ii]][0]), int(particle_properties[col_ind[ii]][1]), 
                                                 particle_properties[col_ind[ii]][2], image_frame, int(particle_properties[col_ind[ii]][0]),
                                                   int(particle_properties[col_ind[ii]][1]), 1, 0, 0]
        
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
            current_particle.loc[0] = [uuid.uuid4(), image_frame, int(particle_properties[ii][0]), int(particle_properties[ii][1]),
                                             particle_properties[ii][2], image_frame, int(particle_properties[ii][0]), 
                                             int(particle_properties[ii][1]), 1, 0, 0]
            
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
                # add to initial and final positions of tracked particle to sqlite database 
                distance = np.sqrt(((row['x_recent']-row['x_init'])**2) + ((row['y_recent']-row['y_init'])**2))
                moving_time = (row['last_frame'] - row['first_frame'])
                if distance == 0:
                    speed = 0
                else: 
                    speed = moving_time/distance

                db.execute(
                    "INSERT INTO trajectories (x_init, y_init, area, x_final, y_final, distance, moving_time, speed, first_frame, last_frame) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (   row['x_init'],
                        row['y_init'],
                        row['area'],
                        row['x_recent'],
                        row['y_recent'],
                        distance, 
                        moving_time,
                        speed,
                        row['first_frame'],
                        row['last_frame']
                        ),
                )

        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: trajectories, id, x_init, y_init, area, x_final, y_final, distance, moving_time, speed, first_frame, last_frame

        # close database
        db.commit()
        db.close()
    
    def kalman_predict(self, statematrix):
        predicted_state = np.dot(self.evolution_matrix, statematrix) #4x1 matrix
        #self.statecovariance = np.add(np.dot(self.evolution_matrix, np.dot(self.state_covariance, self.evolution_matrix.T)), self.process_covariance) #4x4 matrix

        return predicted_state.flatten()

    # #TODO develop updating component of kalman filter
    # def kalman_update(self,... ):

    #     return ....

    def run(self):
        frame_count = 1
        current_frame =1

        images = sorted(self.images)

        #create a pandas dataframe to store images taken in the last second
        recent_df = pd.DataFrame(columns = ['UID', 'first_frame', 'x_init', 'y_init', 'area',
                                                 'last_frame', 'x_recent', 'y_recent', 'count', 'x_velocity', 'y_velocity'])
        
        #start timer
        tic = time.perf_counter()

        #initializing the time of the first image
        time_before = float(images[0].stem)

        #total number of images in the directory
        frame_count_total = len(images)

        print("starting to link the particles together")

        for image_path in images:
            particle_properties = self.extract_particles(self.db_file, image_path, recent_df)

            #saving image time so it doesn't need to be done each time
            img_time = float(image_path.stem)

            #adding data to the data frame if its empty
            if recent_df.empty:
                for pp in range(len(particle_properties)):
                    recent_df.loc[pp] = [uuid.uuid4(), img_time, int(particle_properties[pp][0]), int(particle_properties[pp][1]), particle_properties[pp][2], 
                                    img_time, int(particle_properties[pp][0]), int(particle_properties[pp][1]), 1, 0, 0]
                    
                print(f"{current_frame} was empty")
            
            else: 
                recent_df, cost_matrix = self.linking_particles(recent_df, particle_properties, current_frame, img_time)
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

            time_before = img_time

        #tracking any particles left in the dataframe after finished looking through all the images
        self.track_particles(self.db_file, recent_df)



# need particle's position in every image
# need overall gsd
# need gsd at a given time to know sediment transport rate??

#original code gave
# - bedload transport (g/s)
# - grain size distribution overall (each particle only represented once)
# - grain size distribution at a given point in time
# - particle velocities
# - particle counts
# - grain size distributiosn in terms of D15, D50, D80, etc. 



#removed eccentricity metric from cost matrix as I don't know how to avoid infiinite values when particle is very circular

#TODO ask opinion on thresholding the raw image

