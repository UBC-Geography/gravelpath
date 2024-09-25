## Select the appropriate filters and run them

# particle one at a time and predict movement

## Stochastic method

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


class Particle_Filters:
    def __init__(self, c):
        # defining the config file
        self.c = c

        # writing the path to the sqlite database
        self.db_file = Path(
            c["output"]["path"],
            f'{c["config"]["run_name"]}{c["output"]["file_db_append"]}',
        )

        # selecting and loading which filters need to run
        self.filters = c['particle_linker']['algorithms']

    def run(self):
        for filter in self.filters:
            if filter == "Narrow_Filter":
                from lighttable.particle_linkers.Narrow_Filter import NarrowFilter
                narrow_filter = NarrowFilter(self.c)
                narrow_filter.run()

            if filter == "No_Filter":
                from lighttable.particle_linkers.No_Filter import NoFilter
                no_filter = NoFilter(self.c)
                no_filter.run()

            if filter == "Simple_LAP":
                from lighttable.particle_linkers.Simple_LAP import SimpleLAP
                simple_lap = SimpleLAP(self.c)
                simple_lap.run()

            if filter == "Kalman_Filter":
                from lighttable.particle_linkers.Kalman_Filter import KalmanFilter
                kalman_filter = KalmanFilter(self.c)
                kalman_filter.run()
           