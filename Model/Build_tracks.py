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

import Model.Generate_obstacles as Generate_obstacles
import Model.Plotting as Plotting
from Model.Defaults import *

# Instantiate Classes
plotter = Plotting.Plotting_tracks()
routes = Generate_obstacles.Obstacle_Tracks(MAX_OBS_HEIGHT=14,
                                            MAX_OBS_WIDTH=3,
                                            WINDOW_HEIGHT=25,
                                            WINDOW_WIDTH=9,
                                            N_OBSTABLE_GEN=500,
                                            MIN_GAP=2,
                                            N_TRACKS_GEN=1)

tracks = routes.generate_tracks()

for val, each_matrix in enumerate(tracks):

    name = "Track_New_{}".format(val + 1)

    plotter.plot_grid(matrix=each_matrix,
                      name=name,
                      folder='Track_Img')

    np.save(os.path.join(os.getcwd(),
                         'Track_locations',
                         name), each_matrix)
