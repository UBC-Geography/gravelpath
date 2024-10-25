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

        #saving the run name
        self.run_name = c['config']['run_name']

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

        #create list of values to bin data by
        gsd_bins = [0, 0.5, 0.71, 1, 1.4, 2, 2.83, 4, 5.6, 8, 11.3, 16, 22.6, 32.3, 45, 100]
        
        #bin particles based on their values
        linked_particles['binned'] = pd.cut(linked_particles['grain_size'], bins = gsd_bins, labels = self.phi_fraction, include_lowest=True)

        #create dataframe of zeroes to store particle count and mass
        gsd = pd.DataFrame(np.zeros((len(self.phi_fraction),4)),index=self.phi_fraction,
                                            columns = ['count', 'mass', 'fraction', 'cumulative']) 
        
        for index in gsd.index:
            gsd.loc[index, 'mass'] = linked_particles['mass'][linked_particles['binned'] == index].sum()
            gsd.loc[index, 'count'] = (linked_particles['binned']==index).sum()

        
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

        #Creating columns for the Di dataframe
        Di_cols = []
        for ii in self.Di_percentile:
            Di_cols.append(str(ii))

        #creating array to store Di values
        Di = pd.DataFrame(np.zeros((1,len(Di_cols))), columns = Di_cols, index= [self.run_name])

        #for each target percentile loop through the cumulative fraction
        for m in range(len(Di_cols)):
            for kk in range(gsd.shape[0]-1):
                if (percentages[kk] >= self.Di_percentile[m]) and (percentages[kk + 1] <= self.Di_percentile[m]):
                    Di.loc[self.run_name, Di_cols[m]] = np.exp(np.log(float(phi_frac[kk])) + (np.log(float(phi_frac[kk + 1])) - np.log(float(phi_frac[kk]))) / 
                        (percentages[kk + 1] - percentages[kk]) * (self.Di_percentile[m] - percentages[kk]))

        #logging that the grain size Di calcualtions have started
        logger.info("Grain Size Di calculation complete")

        print(Di)

        return Di
    
    def transport_rate(self, linked_particles, algorithm):

        #logging that sediment transport rate calculations have started
        logger.info(f"Starting to calculate the sediment transport rate for algorithm: {algorithm}")

        #defining the first and last frames
        first_frame = int(np.floor(linked_particles['first_frame'].min()))
        last_frame = int(np.floor(linked_particles['first_frame'].max()))

        #create list of bins to store the data into bins
        transport_bins = np.arange(first_frame, last_frame + 2, 1)

        #bin particles based on their values
        linked_particles['second'] = pd.cut(linked_particles['first_frame'], bins = transport_bins, labels=transport_bins[:-1], include_lowest=True)

        #TODO write the code to also calculate the GSD and Di for the second being examined
        #empty dataframes to store the transport data
        second_transport = np.zeros(last_frame - first_frame + 1)
        minute_transport = np.zeros(int(len(second_transport)/60 + 1))

        #loops through the particles and sums the mass of transported material for each second
        for index, second in enumerate(transport_bins[:-1]):
            second_transport[index] = linked_particles['mass'][linked_particles['second'] == second].sum()

        #list of indexes to split the second transport rate by
        min_index = np.arange(59, last_frame-first_frame + 1, 60)

        #splits the second transport into minutes and then averages those
        for index, array in enumerate(np.split(second_transport, min_index)):
            minute_transport[index] = np.mean(array)

        logger.info("Sediment transport rate calculations complete")

        return second_transport, minute_transport, transport_bins[:-1] - first_frame + 1, np.append(min_index, last_frame - first_frame) + 1
    
    def export_data(self, gsd, Di, instant_rate, moving_avg, seconds, minutes, algorithm):
        
        #export gsd csv
        gsd.to_csv(f"{self.c['output']['path']}/{algorithm}_gsd.csv")

        #export Di csv
        Di.to_csv(f"{self.c['output']['path']}/{algorithm}_Di.csv")

        #export the sediment transport rate
        second_transport = pd.DataFrame({'second': seconds, 'Mass (g)': instant_rate})
        minute_transport = pd.DataFrame({'minute': minutes/60, 'Mass (g)': moving_avg})
        second_transport.to_csv(f"{self.c['output']['path']}/{algorithm}_second_transport.csv")
        minute_transport.to_csv(f"{self.c['output']['path']}/{algorithm}_minute_transport.csv")

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
        plt.plot(seconds, instant_rate, color = 'blue', label = 'Instantaneous Sediment Transport Rate')
        plt.plot(minutes, moving_avg, color = 'r', label = 'Moving Average')
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
            instant_rate, moving_avg, seconds, minutes = self.transport_rate(linked_particles, algorithm)

            #export the data
            self.export_data(gsd, Di, instant_rate, moving_avg, seconds, minutes, algorithm)

            print(f"Finished processing {algorithm} data")