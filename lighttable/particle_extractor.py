# Description: This script is used to process extract the tracked particles fromm processed light table images
# and calculate the grain size distributiono of the particles as well as the transport rate
# 2024-05-08: Sol Leader-Cole, initial version

# numerical processing
import numpy as np
import pandas as pd

#image importing
from pathlib import Path

#databse
import sqlite3

#for plotting
import matplotlib.pyplot as plt

class Particle_Extractor: 
    def __init__(self, c):

        #defining the config file
        self.c = c

        #writing the path to the sqlite database
        self.db_file = Path(
            c["output"]["path"],
            f'{c["config"]["run_name"]}{c["output"]["file_db_append"]}',
        )

        #assigning variable of sediment density in g/mm^3
        self.sediment_density = c['sediment']['density']

        #looking at images to determine the start and end of image recording
        images = list(Path(c["images"]["path"]).rglob("*.tif"))
        images = sorted(images)
        self.time_start = np.floor(float(images[0].stem))
        self.time_end = np.floor(float(images[-1].stem))

    def extract_data(self, db_file):
        
        #connecting to the database
        db = sqlite3.connect(db_file)

        cur = db.cursor()
        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: trajectories, id, x_init, y_init, area, x_final, y_final, distance, moving_time, speed, first_frame, last_frame

        
        #extract all trajectory information from the trajectories table
        cur.execute(f"SELECT x_init, y_init, area, x_final, y_final, distance, moving_time, speed, first_frame, last_frame FROM trajectories")

        #save the information to variable trajectories
        trajectories = cur.fetchall()

        #empty list to save dictionary of values and labels
        data_list=[]

        for particle in trajectories: 
            #insert information into a list of dictionaries
            data_list.append(dict((label,particle[index]) for index,label in enumerate(['x_init', 'y_init', 'area', 'x_final', 'y_final',
                                                                                        'distance','moving_time', 'speed', 'first_frame', 'last_frame'])))
            
        
        #convert the list of dictionaries into a df as it will be easier to manipulate
        linked_particles = pd.DataFrame(data_list, columns=['x_init', 'y_init', 'area', 'x_final', 'y_final', 'distance',
                                                         'moving_time', 'speed', 'first_frame', 'last_frame'])

        #sort the information so that it is arranged according to the frame it first appears in
        if not(linked_particles.empty):
            linked_particles.sort_values("first_frame", ascending=True, inplace=True, ignore_index=True)

        linked_particles.to_csv("particles_data.csv")

        return linked_particles
    
    def calc_grain_size(self, linked_particles):
        
        #calculating the equivalent grain size assuming the particle was circular
        linked_particles['grain_size'] = 2*(np.sqrt(linked_particles['area']/np.pi))

        #calculating the equivalent volume of the particle assuming particle is a sphere
        linked_particles['volume'] = ((1/6)*np.pi*(linked_particles['grain_size']**3))

        #calculating the particles mass assuming constant density
        linked_particles['mass'] = linked_particles['volume'] * self.sediment_density

        return linked_particles

    def calc_gsd(self, linked_particles):
        
        #create dataframe of zeroes to store particle count and mass
        gsd = pd.DataFrame(np.zeros((4,15)),index=['count', 'mass', 'fraction', 'cumulative'], columns=['pan', '0.5', '0.71', '1', '1.4', '2', '2.83',
                                                                               '4', '5.6', '8', '11.3', '16', '22.6', '32.3', '45'])        

        #loop through particles and sort them into the gsd bins
        for index in range(len(linked_particles)):
            grain_size = linked_particles.loc[index]['grain_size']
            grain_mass = linked_particles.loc[index]['mass']

            if grain_size < 0.5:
                gsd.loc['count','pan'] += 1 
                gsd.loc['mass','pan'] += grain_mass

            elif grain_size < 0.71:
                gsd.loc['count','0.5'] += 1 
                gsd.loc['mass','0.5'] += grain_mass

            elif grain_size < 1:
                gsd.loc['count','0.71'] += 1 
                gsd.loc['mass','0.71'] += grain_mass
            
            elif grain_size < 1.4:
                gsd.loc['count','1'] += 1 
                gsd.loc['mass','1'] += grain_mass

            elif grain_size < 2:
                gsd.loc['count','1.4'] += 1 
                gsd.loc['mass','1.4'] += grain_mass

            elif grain_size < 2.83:
                gsd.loc['count','2'] += 1 
                gsd.loc['mass','2'] += grain_mass

            elif grain_size < 4:
                gsd.loc['count','2.83'] += 1 
                gsd.loc['mass','2.83'] += grain_mass

            elif grain_size < 5.6:
                gsd.loc['count','4'] += 1 
                gsd.loc['mass','4'] += grain_mass

            elif grain_size < 8:
                gsd.loc['count','5.6'] += 1 
                gsd.loc['mass','5.6'] += grain_mass

            elif grain_size < 11.3:
                gsd.loc['count','8'] += 1 
                gsd.loc['mass','8'] += grain_mass

            elif grain_size < 16:
                gsd.loc['count','11.3'] += 1 
                gsd.loc['mass','11.3'] += grain_mass

            elif grain_size < 22.6:
                gsd.loc['count','16'] += 1 
                gsd.loc['mass','16'] += grain_mass

            elif grain_size < 32.3:
                gsd.loc['count','22.6'] += 1 
                gsd.loc['mass','22.6'] += grain_mass

            elif grain_size < 45:
                gsd.loc['count','32.3'] += 1 
                gsd.loc['mass','32.3'] += grain_mass

            else:
                gsd.loc['count','45'] += 1 
                gsd.loc['mass','45'] += grain_mass

        #finding the total mass of all transported material
        total_mass = gsd.loc['mass'].sum()

        #initializing value for looping
        prev_col = 'pan'

        #calculating the fraction of each grain size
        for col in gsd.columns:               
            gsd.loc['fraction', col] = gsd.loc['mass', col] / total_mass
            gsd.loc['cumulative', col] = gsd.loc['fraction', col] + gsd.loc['cumulative', prev_col]
            prev_col = col

        return gsd
    
    def cacl_Di(self, gsd):


        #return Di_df
        return None
    
    def transport_rate(self, linked_particles):
        
        #creating list of time stamps
        times = np.arange(self.time_start, self.time_end + 1, 1)

        #initialize list to store the moving average & instantaneous transport rate
        moving_avg = []
        instant_rate = np.zeros_like(times)

        for ii, time_stamp in enumerate(times):
            #get index of each particle who first appears in current second
            time_stamp_index = linked_particles.index[np.floor(linked_particles['first_frame']) == time_stamp]

            #adding information to the instant rate array
            for index in time_stamp_index:
                instant_rate[ii] += linked_particles.loc[index]['mass']

        return instant_rate, moving_avg, times
    
    def export_data(self, gsd, instant_rate, moving_avg, times):
        
        #export gsd csv
        gsd.to_csv(f"{self.c['output']['path']}/gsd.csv")

        #export transport rate csv
        # instant_rate = np.array(instant_rate)
        transport_rate = np.stack((times, instant_rate), axis=1)
        np.savetxt(f"{self.c['output']['path']}/transport_rate.csv", transport_rate)

        #plotting gsd histogram
        plt.bar(gsd.columns, gsd.loc['fraction'])
        plt.ylabel("Fraction")
        plt.xlabel("Retaining Sieve Size")
        plt.title(f"Grain Size Distribution of {self.c['config']['run_name']}")
        plt.savefig(f"{self.c['output']['path']}/grain_size_distribution.pdf")
        plt.clf()
        
        #plotting cumulative gsd histogram
        plt.plot(gsd.columns[1:], gsd.loc['cumulative'][1:])
        plt.ylabel("Fraction")
        plt.xlabel("Retaining Sieve Size")
        plt.title(f"Cumulative Grain Size Distribution of {self.c['config']['run_name']}")
        plt.savefig(f"{self.c['output']['path']}/cumulative_grain_size_distribution.pdf")
        plt.clf()


        #plotting the sediment transport rate
        plt.plot(instant_rate, color = 'blue', label = 'Instantaneous Sediment Transport Rate')
        plt.plot(moving_avg, color = 'r', label = 'Moving Average')
        plt.ylabel("Sediment Transport Rate (g/s)")
        plt.yscale("log")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.title(f"{self.c['config']['run_name']} Sediment Transport")
        plt.savefig(f"{self.c['output']['path']}/sediment_transport_rate.pdf", )
        plt.clf()

    def run(self):
        
        #extracting the data from the sqlite database
        linked_particles = self.extract_data(self.db_file)
        
        #calculating the grain size 
        print("Starting to calculate the grain sizes")
        linked_particles = self.calc_grain_size(linked_particles)
   
        #bin sediment into grain sizes
        gsd = self.calc_gsd(linked_particles)

        #calculating sediment transport rate
        print("Starting to calculate the sediment transport rate")
        instant_rate, moving_avg, times = self.transport_rate(linked_particles)

        Di_df = self.cacl_Di(gsd)

        #TODO save csv of grain size distribution & D50, D90, etc. 
        self.export_data(gsd, instant_rate, moving_avg, times)

        #TODO figure out how to calculate D90, D84, etc. from the GSD