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

#for logging progress
import logging
import time

logger = logging.getLogger(__name__)

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

        #loading phi fractions of sediment
        self.phi_fraction = c['sediment']['classes']

        # selecting and loading which filters were run
        self.filters = c['particle_linker']['algorithms']

        #loading the target percentiles
        self.Di_percentile = c['sediment']['percentiles']

    def extract_data(self, db_file, algorithm):
        
        #logging that processing started
        logger.info(f"Extract data for algorithm: {algorithm}")

        #connecting to the database
        db = sqlite3.connect(db_file)

        cur = db.cursor()
        # Database Configuration
        # Table: particles -> id, image, time, x, y, width, height, area
        # Table: images -> id, image, time, particles
        # Table: seconds -> id, time, particles
        # Table: XX_Filter, id, x_init, y_init, area, x_final, y_final, first_frame, last_frame

        
        #extract all trajectory information from the specific algorithm's table
        cur.execute(f"SELECT x_init, y_init, area, x_final, y_final, first_frame, last_frame FROM {algorithm}")

        #save the information to variable trajectories
        trajectories = cur.fetchall()

        #logging number of particles found
        logger.info(f"Data extract for {algorithm} complete. {len(trajectories)} particles found.")

        #empty list to save dictionary of values and labels
        data_list=[]

        for particle in trajectories: 
            #insert information into a list of dictionaries
            data_list.append(dict((label,particle[index]) for index,label in enumerate(['x_init', 'y_init', 'area', 'x_final', 'y_final',
                                                                                        'first_frame', 'last_frame'])))
            
        
        #convert the list of dictionaries into a df as it will be easier to manipulate
        linked_particles = pd.DataFrame(data_list, columns=['x_init', 'y_init', 'area', 'x_final', 'y_final', 'first_frame', 'last_frame'])

        #sort the information so that it is arranged according to the frame it first appears in
        if not(linked_particles.empty):
            linked_particles.sort_values("first_frame", ascending=False, inplace=True)

        return linked_particles
    
    def calc_grain_size(self, linked_particles, algorithm):
        
        #logging that processing started
        logger.info(f"Calculating grain size for algorithm: {algorithm}")

        #calculating the equivalent grain size assuming the particle was circular
        linked_particles['grain_size'] = 2*(np.sqrt(linked_particles['area']/np.pi))

        #calculating the equivalent volume of the particle assuming particle is a sphere
        linked_particles['volume'] = ((1/6)*np.pi*(linked_particles['grain_size']**3))

        #calculating the particles mass assuming constant density
        linked_particles['mass'] = linked_particles['volume'] * self.sediment_density

        #logging that grain size calculation is finished
        logger.info("Grain size calculation complete.")

        return linked_particles

    def calc_gsd(self, linked_particles, algorithm):
        
        #logging that gsd calculation has started
        logger.info(f"Calculating the grain size distribution for algorithm: {algorithm}")

        #create dataframe of zeroes to store particle count and mass
        gsd = pd.DataFrame(np.zeros((len(self.phi_fraction),4)),index=self.phi_fraction,
                                            columns = ['count', 'mass', 'fraction', 'cumulative'])        

        #loop through particles and sort them into the gsd bins
        for index in range(len(linked_particles)):
            grain_size = linked_particles.loc[index]['grain_size']
            grain_mass = linked_particles.loc[index]['mass']

            if grain_size < 0.5:
                gsd.loc['0.5','count'] += 1 
                gsd.loc['0.5','mass'] += grain_mass

            elif grain_size < 0.71:
                gsd.loc['0.71','count'] += 1 
                gsd.loc['0.71','mass'] += grain_mass

            elif grain_size < 1:
                gsd.loc['1.0','count'] += 1 
                gsd.loc['1.0','mass'] += grain_mass
            
            elif grain_size < 1.4:
                gsd.loc['1.4','count'] += 1 
                gsd.loc['1.4','mass'] += grain_mass

            elif grain_size < 2:
                gsd.loc['2.0','count'] += 1 
                gsd.loc['2.0','mass'] += grain_mass

            elif grain_size < 2.83:
                gsd.loc['2.83','count'] += 1 
                gsd.loc['2.83','mass'] += grain_mass

            elif grain_size < 4:
                gsd.loc['4.0','count'] += 1 
                gsd.loc['4.0','mass'] += grain_mass

            elif grain_size < 5.6:
                gsd.loc['5.6','count'] += 1 
                gsd.loc['5.6','mass'] += grain_mass

            elif grain_size < 8:
                gsd.loc['8.0','count'] += 1 
                gsd.loc['8.0','mass'] += grain_mass

            elif grain_size < 11.3:
                gsd.loc['11.3','count'] += 1 
                gsd.loc['11.3','mass'] += grain_mass

            elif grain_size < 16:
                gsd.loc['16.0','count'] += 1 
                gsd.loc['16.0','mass'] += grain_mass

            elif grain_size < 22.6:
                gsd.loc['22.6','count'] += 1 
                gsd.loc['22.6','mass'] += grain_mass

            elif grain_size < 32.3:
                gsd.loc['32.3','count'] += 1 
                gsd.loc['32.3','mass'] += grain_mass

            elif grain_size < 45:
                gsd.loc['45.0','count'] += 1 
                gsd.loc['45.0','mass'] += grain_mass

            else:
                gsd.loc['64.0','count'] += 1 
                gsd.loc['64.0','mass'] += grain_mass

        #finding the total mass of all transported material
        total_mass = gsd['mass'].sum()

        #initializing value for looping
        prev_ind = '0.5'

        #calculating the fraction of each grain size
        for ind in gsd.index:               
            gsd.loc[ind, 'fraction'] = gsd.loc[ind, 'mass'] / total_mass
            gsd.loc[ind, 'cumulative'] = gsd.loc[ind, 'fraction'] + gsd.loc[prev_ind, 'cumulative']
            prev_ind = ind

        #logging that gsd calculation is complete
        logger.info("Grain size distribution calculation complete.")

        return gsd
    
    def calc_Di(self, gsd, algorithm):
         
        #logging that the grain size Di calcualtions have started
        logger.info(f"Starting to calculate the grain size Di for algorithm: {algorithm}")

        #creating array to store percentage values
        percentages = np.zeros(len(gsd))

        #store cumulative fraction percentages in descending order
        percentages = gsd['cumulative'][::-1] * 100

        #store phi fractions in descending order
        phi_frac = self.phi_fraction[::-1]

        print(percentages)

        #Creating columns for the Di dataframe
        Di_cols = []
        for ii in self.Di_percentile:
            Di_cols.append(str(ii))

        #creating array to store Di values
        Di = pd.DataFrame(np.zeros((1,len(Di_cols))), columns = Di_cols)

        #for each target percentile loop through the cumulative fraction
        for m in range(len(Di_cols)):
            for kk in range(gsd.shape[0]-1):
                if (percentages[kk] >= self.Di_percentile[m]) and (percentages[kk + 1] <= self.Di_percentile[m]):
                    print(f"target percentile {self.Di_percentile[m]}")
                    print(percentages[kk])
                    print(Di_cols[m])
                    Di.loc[0, Di_cols[m]] = np.exp(np.log(float(phi_frac[kk])) + (np.log(float(phi_frac[kk + 1])) - np.log(float(phi_frac[kk]))) / 
                        (percentages[kk + 1] - percentages[kk]) * (self.Di_percentile[m] - percentages[kk]))

        print(Di)

        #logging that the grain size Di calcualtions have started
        logger.info("Grain Size Di calculation complete")

        return Di
    
    def transport_rate(self, linked_particles, algorithm):
        
        #logging that sediment transport rate calculations have started
        logger.info(f"Starting to calculate the sediment transport rate for algorithm: {algorithm}")

        #initialize list to store the moving average & instantaneous transport rate
        moving_avg = []
        instant_rate = []

        #finding the time in seconds of the first frame
        time_stamp = np.floor(linked_particles.loc[0]['first_frame'])

        #initialising value to keep current transport rate
        current_rate = 0

        for index in range(len(linked_particles)):
            if np.floor(linked_particles.loc[index]['first_frame']) == time_stamp:
                #adding information to the transport rate
                current_rate += linked_particles.loc[index]['mass']

            else: 
                #adding info to the lists
                instant_rate.append(current_rate)
                moving_avg.append(np.mean(instant_rate))

                #updating time stamp for the new second
                time_stamp = np.floor(linked_particles.loc[index]['first_frame'])

                #adding data to the transport rate
                current_rate = linked_particles.loc[index]['mass']

        #adding final information to the lists
        instant_rate.append(current_rate)
        moving_avg.append(np.mean(instant_rate))

        logger.info("Sediment transport rate calculations complete")

        return instant_rate, moving_avg
    
    def export_data(self, gsd, Di, instant_rate, moving_avg, algorithm):
        
        #export gsd csv
        gsd.to_csv(f"{self.c['output']['path']}/{algorithm}_gsd.csv")

        #export Di csv
        Di.to_csv(f"{self.c['output']['path']}/{algorithm}_Di.csv")

        #plotting gsd histogram
        plt.bar(gsd.index, gsd['fraction'])
        plt.ylabel("Fraction")
        plt.xlabel("Retaining Sieve Size")
        plt.title(f"Grain Size Distribution of {self.c['config']['run_name']} ({algorithm})")
        plt.savefig(f"{self.c['output']['path']}/{algorithm}_grain_size_distribution.pdf")
        plt.clf()
        
        #plotting cumulative gsd histogram
        plt.plot(gsd.index, gsd['cumulative'])
        plt.ylabel("Fraction")
        plt.xlabel("Retaining Sieve Size")
        plt.title(f"Cumulative Grain Size Distribution of {self.c['config']['run_name']} ({algorithm})")
        plt.savefig(f"{self.c['output']['path']}/{algorithm}_cumulative_grain_size_distribution.pdf")
        plt.clf()


        #plotting the sediment transport rate
        plt.plot(instant_rate, color = 'blue', label = 'Instantaneous Sediment Transport Rate')
        plt.plot(moving_avg, color = 'r', label = 'Moving Average')
        plt.ylabel("Sediment Transport Rate (g/s)")
        plt.yscale("log")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.title(f"{self.c['config']['run_name']} Sediment Tansport Rate ({algorithm})")
        plt.savefig(f"{self.c['output']['path']}/{algorithm}_sediment_transport_rate.pdf", )
        plt.clf()

    def run(self):
        
        for algorithm in self.filters:

            #extracting the data from the sqlite database
            linked_particles = self.extract_data(self.db_file, algorithm)
            
            #calculating the grain size 
            linked_particles = self.calc_grain_size(linked_particles, algorithm)
    
            #bin sediment into grain sizes
            gsd = self.calc_gsd(linked_particles, algorithm)

            #calculating the grain size Di
            Di = self.calc_Di(gsd, algorithm)

            #calculating sediment transport rate
            instant_rate, moving_avg = self.transport_rate(linked_particles, algorithm)

            #TODO save csv of grain size distribution & D50, D90, etc. 
            self.export_data(gsd, Di, instant_rate, moving_avg, algorithm)

            print(f"Finished processing {algorithm} data")



        #TODO figure out how to calculate D90, D84, etc. from the GSD