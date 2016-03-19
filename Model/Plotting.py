# Purpose: Simple Image Plotting of Numpy Matrices
#
#   Info: A set of of functions for simple plotting in Matplotlib
#           - include example plot function to inspect functionality
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Plotting

matplotlib.style.use('ggplot')


class Plotting_tracks(object):
    """

    Simple Image Plotting of Numpy Matrices

    """

    def __init__(self):
        logging.debug("Loaded Plotting Function")

    @staticmethod
    def example():
        """ To run the example on Command Line:

        Input
        #####
            import Plotting
            Plotting_track().example()

        Output
        ######
            Saved plot in directory
            Plot Show in new window

        """
        fig, ax = plt.subplots()

        image = np.random.uniform(size=(10,
                                        10))  # Plotting Matrix

        ax.imshow(image,
                  cmap=plt.get_cmap('RdGy'),
                  interpolation='nearest')

        # Set the title
        ax.set_title('Example Grid')

        # Move left and bottom spines outward by 10 points
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        name = 'plot_example.png'
        plt.savefig(os.path.join(os.getcwd(),
                                 'Model',
                                 'Obstacle_Img',
                                 name))
        plt.show()

    @staticmethod
    def plot_grid(matrix, name, folder):
        """ Plot Grid function will attempt to plot a matrix

        Input
        #####
            numpy matrix

        Output
        ######
            Saved plot in directory
            Plot Show in new window

        """
        if isinstance(matrix, np.ndarray):
            plt.ioff()
            fig, ax = plt.subplots()
            image = matrix
            ax.imshow(image,
                      cmap=plt.get_cmap('gnuplot'),
                      interpolation='nearest')

            # Set the title
            ax.set_title('Plot of {}'.format(name))

            # Move left and bottom spines outward by 10 points
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            # Save and Show Plot
            name = name + '.png'
            plt.savefig(os.path.join(os.getcwd(),
                                     'Model',
                                     folder,
                                     name))

            return name

        else:
            logging.error("Data provided was not a Matrix")
            return 'Error'


class plotting_model(object):
    """

    Plotting the Q-Matrix from model [1, 2]

    """

    def __init__(self):
        self.DF = None
        self.Q_Matrix = None
        self.DF_new = None
        self.Q = None

    def get_q_matrix(self, model_q=None, nb_actions=None):
        """
        Function to Generate the Reward Matrix

        :param model_q: dict
        :param nb_actions: int
        :return: None
        """
        assert isinstance(model_q, dict) and isinstance(nb_actions, int), \
            "Object Types not as Expected"

        self.Q = model_q
        length = len(model_q)
        splitting_keys = list(model_q)
        self.Q_Matrix = np.zeros((length, nb_actions))

        for val, key in enumerate(splitting_keys):
            self.Q_Matrix[val][key[1]] = model_q[key]

    def plot_q_matrix(self, f_name):
        """
        Plot the R-Matrix

        :param f_name: str
        :return: None (saves to folder)
        """
        assert self.Q_Matrix is not None, \
            "Call get_q_matrix before using this function"

        plotter = Plotting.Plotting_tracks()
        plotter.plot_grid(matrix=self.Q_Matrix,
                          name=f_name,
                          folder='Q_Matrix_Plots')

    def get_details(self, ):
        """
        Get basic metrics about the States

        :return: dict
        """
        assert self.Q is not None, \
            "Call get_q_matrix before using this function"

        min_q = min(self.Q.values())
        max_q = max(self.Q.values())
        data = self.Q.values()
        q_data = dict(min=min_q,
                      max=max_q,
                      data=data)

        return q_data
