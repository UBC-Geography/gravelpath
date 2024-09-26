#NoFilter
# simple stores the particle properties of every particle from all the images

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

logger = logging.getLogger(__name__)

class NoFilter:
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
        db.execute("DROP TABLE IF EXISTS no_filter")

        # create table to append trajectory data per particle
        db.execute(
            "CREATE TABLE no_filter (id INTEGER PRIMARY KEY, x_init REAL, y_init REAL, area REAL, x_final REAL, y_final REAL, first_frame REAL, last_frame REAL)"
        )

        #close database
        db.commit()
        db.close()

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
    
    def track_particles(self, db_file, to_track):
        
        #connecting to the database
        db = sqlite3.connect(db_file)

        if not(to_track.empty):
            for index, row in to_track.iterrows():
                db.execute(
                    "INSERT INTO no_filter (x_init, y_init, area, x_final, y_final, first_frame, last_frame) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (   row['x_init'],
                        row['y_init'],
                        row['area'],
                        row['x_recent'],
                        row['y_recent'],
                        row['first_frame'],
                        row['last_frame']
                        ),
                )

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
        current_frame =1

        images = sorted(self.images)

        #create a pandas dataframe to store images taken in the last second
        recent_df = pd.DataFrame(columns = ['UID',
                                            'first_frame',
                                            'x_init', 
                                            'y_init', 
                                            'area',
                                            'last_frame', 
                                            'x_recent', 
                                            'y_recent', 
                                            'count'])
        
        #start timer
        tic = time.perf_counter()

        #initializing the time of the first image
        time_before = float(images[0].stem)

        #total number of images in the directory
        frame_count_total = len(images)

        print("starting to link the particles together")

        for image_path in images:
            
            #saving image time so it doesn't need to be done each time
            img_time = float(image_path.stem)
            
            particle_properties = self.extract_particles(self.db_file, image_path)

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
                    
            self.track_particles(self.db_file, recent_df)
            
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