# Purpose: Function Will Create Tracks
#
#   Info: A track is defined as a series of obstacles
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#
import os
import sys

import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import Plotting
import Generate_obstacles
from Defaults import *

# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.INFO)

# Instantiate Classes
logging.info("Loading Plotting and Obstacle Generation Function")
plotter = Plotting.Plotting_tracks()
routes = Generate_obstacles.Obstacle_Tracks(MAX_OBS_HEIGHT=14,
                                            MAX_OBS_WIDTH=3,
                                            WINDOW_HEIGHT=25,
                                            WINDOW_WIDTH=9,
                                            N_OBSTABLE_GEN=500,
                                            MIN_GAP=2,
                                            N_TRACKS_GEN=1)

logging.info("Generate Tracks / Paths")
tracks = routes.generate_tracks()

logging.info("Plot and Save Obstacles")
for val, each_matrix in enumerate(tracks):
    logging.debug("Prepare Track Name")
    name = "Track_{}".format(val + 1)

    logging.debug("Plotting Matrix")
    plotter.plot_grid(matrix=each_matrix,
                      name=name,
                      folder='Track_Img')

    logging.debug("Saving Matrix")
    np.save(os.path.join(os.getcwd(),
                         'Track_locations',
                         name), each_matrix)
    logging.debug("Matrix Saved")
