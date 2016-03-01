# Purpose: Main Function - Training and Plotting Results
#   Info: Change the Parameters at the top of the scrip to change how the Agent interacts
#   Developed as part of the Software Agents Course at City University
#   Dev: Dan Dixey and Enrico Lopedoto

import sys
import os
from time import time
import logging
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from Helicopter import helicopter
import World as W
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import numpy.ma as ma
matplotlib.style.use('ggplot')

# Logging Controls Level of Printing
logging.basicConfig(
    format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
    level=logging.DEBUG)

logging.info("Setting Parameters:")
trials = 50                                   # Number of Trials to Train the Agent on
# List of Rewards [Completed, Crashed, Open]
r_values = [300, -100, 10]
p_values = [0.3, 0.5, 0.3]
d_par = 10
par = "epsilon"  # epsilon, alhpa, gamma
plt_factor = [0, list(range(trials - 0, trials + 1)),
              0.10]   # Plot first X items
# Number of Possible Actions Helicopter can take
nb_actions = 3
model = [1, 2]                                   # Model Selection
# Learning Rate Decay (if model=2)
decay_rate = 0.9
# Learning Rate Decay update rate (if model=2)
nb_action_change = 6000
# Temporal Difference value (if model=3)
lmb = 2
color = ["red", "blue"]
moddef = ["Random", "Epsylon Decay"]
# file_name=None - Loads a Randomly Generated Track

logging.info("Starting the Learning Process")
for z in model:
    logging.info("Define Real-time Plotting Figure")
    HeliWorld = W.helicopter_world(file_name="Track_1.npy")

    fig = plt.figure()
    fig.canvas.draw()
    plt.subplot(2, 2, 1)
    plt.title('Best Realized Patter', fontsize=10)
    plt.xlabel('Track Length', fontsize=8)
    plt.ylabel('Track Width', fontsize=8)
    my_axis = plt.gca()
    my_axis.set_xlim(0, HeliWorld.track_width)
    my_axis.set_ylim(0, HeliWorld.track_height)
    im1 = plt.imshow(HeliWorld.track,
                     cmap=plt.cm.jet,
                     interpolation='nearest',
                     vmin=-1,
                     vmax=6)
    plt.colorbar(im1, fraction=0.01, pad=0.01)

    plt.subplot(2, 2, 2)
    plt.title('Final Q Matrix', fontsize=10)
    plt.xlabel('Track Length', fontsize=8)
    plt.ylabel('Track Width', fontsize=8)
    my_axis = plt.gca()
    my_axis.set_xlim(0, HeliWorld.track_width)
    my_axis.set_ylim(0, HeliWorld.track_height)
    a = np.zeros(shape=(HeliWorld.track_height, HeliWorld.track_width))
    im2 = plt.imshow(a)
    plt.colorbar(im2, fraction=0.01, pad=0.01)

    plt.subplot(2, 2, 3)
    plt.title('Completion Time Chart', fontsize=10)
    plt.xlabel('Trial Numbers', fontsize=8)
    plt.ylabel('Seconds Per Trial', fontsize=8)
    my_axis = plt.gca()
    my_axis.set_xlim(0, trials)

    plt.subplot(2, 2, 4)
    plt.title('Learning Chart', fontsize=10)
    plt.xlabel('Trial Numbers', fontsize=8)
    plt.ylabel('End Location', fontsize=8)
    my_axis = plt.gca()
    my_axis.set_xlim(0, trials)
    my_axis.set_ylim(0, 1)

    colors = cm.rainbow(np.linspace(0, 1, d_par))
    for aa in range(1, d_par + 1):
        param = float(aa / 10)
        p_value = [0, 0, 0]
        if par == "epsilon":
            p_value[0] = param
        else:
            p_value[0] = p_values[0]
        if par == "alpha":
            p_value[1] = param
        else:
            p_value[1] = p_values[1]
        if par == "gamma":
            p_value[2] = param
        else:
            p_value[2] = p_values[2]

        logging.info("Load Helicopter and World")
        HeliWorld = W.helicopter_world(file_name="Track_1.npy")
        Helicopter1 = helicopter(world=HeliWorld,
                                 n_action=nb_actions,
                                 reward_values=r_values,
                                 parameters=p_value,
                                 decay=decay_rate,
                                 rate=nb_action_change,
                                 model=z)

        st = time()
        time_metrics = []
        while HeliWorld.trials <= trials:
            if HeliWorld.trials % plt_factor[2] == 0 and HeliWorld.trials > 0:
                rate = ((time() - st + 0.01) / HeliWorld.trials)
                value = [HeliWorld.trials, rate]
                time_metrics.append(value)
                logging.info(
                    "Trials Completed: {} at {:.4f} Trails / seconds".format(value[0], value[1]))

            while True:
                output = Helicopter1.update()
                if not output:
                    Helicopter1.reset()
                    rate = ((time() - st + 0.01) / HeliWorld.trials)
                    value = [HeliWorld.trials,
                             rate]
                    plt.subplot(2, 2, 4)
                    plt.scatter(HeliWorld.trials,
                                Helicopter1.final_location[-1][0] /
                                float(HeliWorld.track_width),
                                s=np.pi * (1 * (1))**2,
                                c=colors[aa - 1],
                                alpha=0.5)
                    plt.legend()

                    plt.subplot(2, 2, 3)
                    plt.scatter(value[0],
                                value[1],
                                s=np.pi * (1 * (1))**2,
                                c=colors[aa - 1],
                                alpha=0.5)

                    break

                if HeliWorld.trials <= plt_factor[
                        0] or HeliWorld.trials in plt_factor[1]:
                    # Primary Title
                    rate = (time() - st + 0.01) / HeliWorld.trials
                    value = [HeliWorld.trials,
                             rate]
                    fig.suptitle(
                        'Time for Trial Completion: {} - \
                                  Current State: {} - Current Location: {}\n\
                                  Trials Completed: {} with Total Time {:.3f} seconds\n\
                                  Agent Model: {} \n\
                                  Agent Parameters: \
                                  alpha {} - epsilon {:.4f} - gamma {} - Number of Actions: {}\n\
                                  World Paramers: \
                                  Length of Track: {} - Width of Track: {}'.format(
                            HeliWorld.trials,
                            Helicopter1.current_state,
                            Helicopter1.current_location,
                            value[0],
                            time() - st + 0.01,
                            moddef[z - 1],
                            Helicopter1.ai.alpha,
                            Helicopter1.ai.epsilon,
                            Helicopter1.ai.gamma,
                            nb_actions,
                            HeliWorld.track_width,
                            HeliWorld.track_height),
                        fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='top')

                    # Plotting Real-time plot
                    plt.subplot(2, 2, 1)
                    plt.imshow(HeliWorld.track,
                               cmap=plt.cm.jet,
                               interpolation='nearest',
                               vmin=-1,
                               vmax=6)

                    plt.scatter(Helicopter1.current_location[0],
                                Helicopter1.current_location[1],
                                s=np.pi * (1 * (1))**2,
                                c=color[model[z - 1] - 1],
                                alpha=0.5)

                view_current = Helicopter1.q_matrix[
                    Helicopter1.current_location[0] - 1][0]
                qw_mat = []
                for i in range(5):
                    key = (view_current, i + 1)
                    if key not in list(Helicopter1.ai.q.keys()):
                        qw_mat.append(0)
                    else:
                        qw_mat.append(Helicopter1.ai.q[key])
                m = Helicopter1.current_location[0]
                start = int(Helicopter1.current_location[1])
                bigger_array = np.zeros(shape=(1, HeliWorld.track_height + 3))
                smaller_array = np.array(qw_mat)
                masked_smaller_array = ma.masked_array(smaller_array, mask=[5])

                if len(bigger_array[0, start - 2:start + 3]) < 5:
                    bigger_array[
                        0, max(start - 2, 0):max(start + 3, 5)] = masked_smaller_array
                else:
                    bigger_array[0, start - 2:start + 3] = masked_smaller_array
                a[:, m - 1] += bigger_array[0, :HeliWorld.track_height]
            plt.subplot(2, 2, 2)
            plt.imshow(a)
           # plt.pause(1e-10)

            logging.debug('Starting next iteration')
            HeliWorld.trials += 1
        et = time()
    logging.info("Time Taken: {} seconds".format(et - st))

#logging.info("Plotting the Q-Matrix")
#model_plot = plotting_model()
# model_plot.get_q_matrix(model_q=Helicopter1.ai.q,
#                        nb_actions=nb_actions)
# model_plot.plot_q_matrix('Q-Matrix')
