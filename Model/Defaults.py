# Purpose: Stores the Default Values for useful for Generating Random Tracks
#
#   Info: A set of parameters that can be used for quick testing
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#
import logging


# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.INFO)


WINDOW_HEIGHT = 50      # Screen Height
WINDOW_WIDTH = 7       # Screen Width
MAX_OBS_HEIGHT = 23     # Maximum Obstacle Height
MAX_OBS_WIDTH = 2       # Maximum Obstacle Width
N_OBSTABLE_GEN = 20     # Number of Obstacles to Generate
MIN_GAP = 0             # Trimming Factor when collating Obstacles
N_TRACKS_GEN = 2        # Number of Tracks to Generate
logging.info("Loaded Default Track Parameters")
