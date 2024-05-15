import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import toml


con = sqlite3.connect('output/lighttable_test_output.sqlite')

cur = con.cursor()

# cur.execute(f"SELECT x, y FROM particles WHERE (time = '{1/120}')")
cur.execute(f"SELECT x_init, y_init, area, x_final, y_final, distance, moving_time, speed, first_frame, last_frame FROM trajectories")

coords = cur.fetchall()

print(coords)

np.savetxt("trajectories.csv", coords, delimiter=',')

cur.execute(f"SELECT x y FROM particles")

particles = cur.fetchall()
print(len(particles))

trajectories = cur.fetchall()

for particle in trajectories[0:1]: 
    #TODO insert information into a pd database
    print(particle)

len(particle)

particle[10]

linked_particles = pd.DataFrame(columns = ['x_init', 'y_init', 'area', 'x_final', 'y_final', 'distance',
                                                         'moving_time', 'speed', 'first_frame', 'last_frame'])

row_list=[]

for index, particle in enumerate(trajectories): 
    # for ii in range(10):
    #insert information into a pd database
    row_list.append(dict((a,particle[i]) for i,a in enumerate(['x_init', 'y_init', 'area', 'x_final', 'y_final', 'distance',
                                                         'moving_time', 'speed', 'first_frame', 'last_frame'])))

    print(index)
    print(particle)

row_list

df4 = pd.DataFrame(row_list, columns=['x_init', 'y_init', 'area', 'x_final', 'y_final', 'distance',
                                                         'moving_time', 'speed', 'first_frame', 'last_frame'])



df4

df4['grain_size'] = np.sqrt(df4['area']/np.pi)
df4

df4.loc[0]['first_frame']//1

1.5//1



6.47//1

for ii in range(len(df4)):
    print(df4.loc[ii])


gsd = pd.DataFrame(np.zeros((2,15)),index=['count', 'mass'], columns=['pan', '0.5', '0.71', '1', '1.4', '2', '2.83',
                                                                     '4', '5.6', '8', '11.3','16', '22.6', '32.3', '45']) 

gsd

for index in range(len(df4)):
    if df4.loc[index,'grain_size'] < 0.5:
        gsd.loc['count','pan'] += 1 
        gsd.loc['mass']['pan'] += linked_particles.loc[index,'area']

df4


for ii in range(10):
    gsd.loc['mass','16'] += 2

gsd

plt.bar(gsd.columns, gsd.loc['mass'])
plt.savefig('output/test.png')
plt.clf()

df4.to_csv()

gsd.loc['mass', '8'] = 26

gsd.loc['mass'].sum()

gsd.columns[0]

for col in gsd.columns:
    print(col)

config_files = Path("configs_to_run").rglob("*.toml")

# iterate over each config file
for cf in config_files:
    # load the config file
    c = toml.load(cf)

t = list(Path(c['images']['path']).rglob('*.tif'))
t[0].stem

images = list(Path(c["images"]["path"]).rglob("*.tif"))