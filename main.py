# Purpose: Main Function - Training and Plotting Results
#
#   Info: Change the Parameters at the top of the scrip to change how the Agent interacts
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
import sys
import os
from time import time
import logging
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from Helicopter import helicopter
import World as W
from Plotting import plotting_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')


# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.DEBUG)


logging.info("Setting Parameters:")
# Model Settings
settings = dict(trials=200,
                completed=500,
                crashed=-100,
                open=5,
                alpha=0.65,
                epsilon=0.75,
                gamma=0.7,
                nb_actions=5,
                model=4,
                epsilon_decay=0.9,
                epsilon_action=6000,
                lambda_td=0.5)
# Plot Settings
plot_settings = dict(print_up_to=1,
                     end_range=list(range(settings['trials'] - 1,
                                          settings['trials'] + 1)),
                     print_rate=5)

logging.info("Load Helicopter and World")
HeliWorld = W.helicopter_world(file_name="Track_1.npy")
# file_name=None - Loads a Randomly Generated Track
Helicopter1 = helicopter(world=HeliWorld,
                         settings=settings)

logging.info("Define Real-time Plotting Figure")
fig = plt.figure(1)
gs = gridspec.GridSpec(3, 3)
fig.canvas.draw()
plt.subplot(gs[1, :-1])
plt.title('Completion Time Chart', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('Seconds Per Trial', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, settings['trials'])

plt.subplot(gs[-1, :-1])
plt.title('Learning Chart', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('End Location', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, settings['trials'])
my_axis.set_ylim(0, HeliWorld.track_width)

plt.subplot(gs[1, -1:])
plt.title('Field View', fontsize=10)
plt.xlabel('View Depth', fontsize=8)
plt.ylabel('View Height', fontsize=8)

plt.subplot(gs[-1, -1:])
plt.title('Q-Matrix', fontsize=10)
plt.ylabel('State - Action', fontsize=8)

logging.info("Starting the Learning Process")
st = time()
time_metrics = []
while HeliWorld.trials <= settings['trials']:
    # On the Last Trail give the Model full control
    if HeliWorld.trials == settings['trials']:
        Helicopter1.ai.epsilon = 1e-9

    # Print out logging metrics
    if HeliWorld.trials % plot_settings[
            'print_rate'] == 0 and HeliWorld.trials > 0:
        rate = HeliWorld.trials / (time() - st + 0.01)
        value = [HeliWorld.trials, rate]
        time_metrics.append(value)
        logging.info(
            "Trials Completed: {} at {:.4f} seconds per trail".format(
                value[0], value[1]))
    # Inner loop of episodes
    while True:
        output = Helicopter1.update()
        if not output:
            Helicopter1.reset()
            rate = (time() - st + 0.01) / HeliWorld.trials
            value = [HeliWorld.trials,
                     rate]
            plt.subplot(gs[-1, :-1])
            plt.scatter(Helicopter1.final_location[-1][1],
                        Helicopter1.final_location[-1][0],
                        s=np.pi * (2 * 1) ** 2,
                        c="red",
                        alpha=0.5)

            plt.subplot(gs[1, :-1])
            plt.scatter(value[0],
                        value[1],
                        s=np.pi * (2 * 1) ** 2,
                        c="red",
                        alpha=0.5)
            break

        if HeliWorld.trials <= plot_settings[
                'print_up_to'] or HeliWorld.trials in plot_settings['end_range']:
            # Primary Title
            rate = (time() - st + 0.01) / HeliWorld.trials
            value = [HeliWorld.trials,
                     rate]
            fig.suptitle(
                'Time for Trial Completion: {}\n\
                          Current State: {}, Current Location: {}\n\
                          Trials Completed: {} with Total Time {:.3f} seconds\n\
                          Agent Parameters: \n\
                          alpha {}, epsilon {:.5f}, gamma {}, Number of Actions: {}, Agent Algorithm: {} \n\
                          World Paramers: \n\
                          Length of Track: {}, Width of Track: {}'.format(
                    HeliWorld.trials,
                    Helicopter1.current_state,
                    Helicopter1.current_location,
                    value[0],
                    time() - st + 0.01,
                    settings['alpha'],
                    settings['epsilon'],
                    settings['gamma'],
                    settings['nb_actions'],
                    Helicopter1.titles[settings['model'] - 1],
                    HeliWorld.track_width,
                    HeliWorld.track_height),
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='top')
            # Plotting Real-time plot
            plt.subplot(gs[0, :])
            plt.imshow(HeliWorld.track,
                       cmap=plt.cm.rainbow,
                       interpolation='nearest')
            plt.scatter(Helicopter1.current_location[0],
                        Helicopter1.current_location[1],
                        s=np.pi * (2 * 1) ** 2,
                        c="red",
                        alpha=0.5)

            view_current = Helicopter1.q_matrix[-1][1][:-1]
            qw_mat = []
            for i in range(5):
                key = (view_current, i + 1)
                if key not in list(Helicopter1.ai.q.keys()):
                    qw_mat.append(0)
                else:
                    qw_mat.append(Helicopter1.ai.q[key])

            # Field of View Plot
            plt.subplot(gs[1, -1:])
            plt.imshow(np.array(view_current).reshape(5, 3),
                       cmap=plt.cm.rainbow,
                       interpolation='nearest',
                       vmin=-1,
                       vmax=7,
                       extent=(0, 2, 0, 5))

            # Q-Matrix
            plt.subplot(gs[-1, -1:])
            plt.imshow(np.asarray(qw_mat).reshape(5, 1),
                       cmap=plt.cm.rainbow,
                       interpolation='nearest')
            plt.pause(1e-10)

    logging.debug('Starting next iteration')
    HeliWorld.trials += 1

et = time()
logging.info("Time Taken: {} seconds".format(et - st))

if settings['model'] < 4:
    logging.info("Plotting the Q-Matrix")
    model_plot = plotting_model()
    model_plot.get_q_matrix(model_q=Helicopter1.ai.q,
                            nb_actions=settings['nb_actions'])
    model_plot.plot_q_matrix('Q-Matrix')
else:
    name = 'alpha_{}_epsilon_{}_gamma_{}_trails_{}_nb_actions_{}_model_{}'.format(
        settings['alpha'],
        settings['epsilon'],
        settings['gamma'],
        settings['trials'],
        settings['nb_actions'],
        settings['model'])
    Helicopter1.ai.save_model(name=name)
