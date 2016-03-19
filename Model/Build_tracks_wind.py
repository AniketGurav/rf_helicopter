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

import numpy as np

import Model.Plotting as Plotting
import Model.Wind_Generation as Wind_Generation
from Model.Defaults import *


# Instantiate Classes
plotter = Plotting.Plotting_tracks()
routes = Wind_Generation.Obstacle_Tracks(MAX_OBS_HEIGHT=34,
                                         MAX_OBS_WIDTH=7,
                                         WINDOW_HEIGHT=80,
                                         WINDOW_WIDTH=9,
                                         N_OBSTABLE_GEN=100,
                                         MIN_GAP=2,
                                         N_TRACKS_GEN=1)

tracks = routes.generate_tracks

for val, each_matrix in enumerate(tracks):

    name = "Track_Wind_{}".format(val + 3)

    plotter.plot_grid(matrix=each_matrix,
                      name=name,
                      folder='Track_Img')

    np.save(os.path.join(os.getcwd(),
                         'Model',
                         'Track_locations',
                         name), each_matrix)
