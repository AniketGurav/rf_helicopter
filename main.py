# Purpose: Main Function - Training and Plotting Results
#
#   Info: Change the Parameters at the top of the scrip to change how the Agent interacts
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#   Updated: 1/3/2016
#
import logging
import os
import sys
from time import time

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from Model.Helicopter import helicopter
from Model import World as W
from Model.Plotting import plotting_model
import matplotlib.pyplot as plt
import numpy as np
from Settings import *
from Model import Utils
plt.style.use('ggplot')


# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.DEBUG)


logging.info("Setting Parameters:")
# Model Settings
case = 'case_two'
settings_ = case_lookup[case]
iterations, settings = get_indicies(settings_)

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
                 vmax=8)
plt.colorbar(im1, fraction=0.01, pad=0.01)

plt.subplot(2, 2, 2)
plt.title('Final Q Matrix', fontsize=10)
plt.xlabel('Track Length', fontsize=8)
plt.ylabel('Track Width', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, HeliWorld.track_width)
my_axis.set_ylim(0, HeliWorld.track_height)
a = np.zeros(shape=(HeliWorld.track_height,
                    HeliWorld.track_width))
im2 = plt.imshow(a)
plt.colorbar(im2, fraction=0.01, pad=0.01)

plt.subplot(2, 2, 3)
plt.title('Completion Time Chart', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('Seconds Per Trial', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, settings['trials'])

plt.subplot(2, 2, 4)
plt.title('Learning Chart', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('End Location', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, settings['trials'])
my_axis.set_ylim(0, 1)

colors = plt.cm.rainbow(np.linspace(0, 1, iterations))

logging.info("Starting the Learning Process")
st = time()
time_metrics = []

logging.info('Dealing with Model: {}'.format(Utils.titles[settings['model']]))
for value in range(iterations):
    if iterations > 1:
        settings = get_settings(dictionary=settings_,
                                ind=value)
        Helicopter1 = helicopter(world=HeliWorld,
                                 settings=settings)

    while HeliWorld.trials <= settings['trials']:
        # On the Last Trail give the Model full control
        if HeliWorld.trials == settings['trials']:
            Helicopter1.ai.epsilon, settings['epsilon'] = 1e-9, 1e-9

        # Print out logging metrics
        if HeliWorld.trials % plot_settings[
                'print_rate'] == 0 and HeliWorld.trials > 0:
            rate = ((time() - st + 0.01) / HeliWorld.trials)
            value = [HeliWorld.trials, rate]
            time_metrics.append(value)
            logging.info(
                "Trials Completed: {} at {:.4f} seconds / trial".format(value[0], value[1]))

        # Inner loop of episodes
        while True:
            output = Helicopter1.update()
            if not output:
                Helicopter1.reset()
                rate = (time() - st + 0.01) / HeliWorld.trials
                value = [HeliWorld.trials,
                         rate]
                plt.subplot(2, 2, 4)
                plt.scatter(HeliWorld.trials,
                            Helicopter1.final_location[-1][0] /
                            float(HeliWorld.track_width),
                            s=np.pi * (1 * 1) ** 2,
                            c=model_color[settings['model'] - 1],
                            alpha=0.5)
                plt.legend()

                plt.subplot(2, 2, 3)
                plt.scatter(value[0],
                            value[1],
                            s=np.pi * (1 * 1) ** 2,
                            c=colors[settings['model'] - 1],
                            alpha=0.5)

                break

            if HeliWorld.trials <= plot_settings[
                    'print_up_to'] or HeliWorld.trials in plot_settings['end_range']:
                # Primary Title
                rate = (time() - st + 0.01) / HeliWorld.trials
                value = [HeliWorld.trials,
                         rate]
                fig.suptitle(
                    'Time for Trial Completion: {} - \
                                      Current State: {} - Current Location: {}\n\
                                      Trials Completed: {} with Total Time {:.3f} seconds\n\
                                      Agent Model: {} \n\
                                      Agent Parameters: \n\
                                      alpha {} - epsilon {:.4f} - gamma {} - Number of Actions: {}\n\
                                      World Paramers: \
                                      Length of Track: {} - Width of Track: {}'.format(
                        HeliWorld.trials,
                        Helicopter1.current_state,
                        Helicopter1.current_location,
                        value[0],
                        time() - st + 1e-9,
                        Utils.titles[settings['model'] - 1],
                        settings['alpha'],
                        settings['epsilon'],
                        settings['gamma'],
                        settings['nb_actions'],
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
                            s=np.pi * (1 * 1) ** 2,
                            c=colors[settings['model'] - 1],
                            alpha=0.5)

                plt.subplot(2, 2, 2)
                plt.imshow(a)
                plt.pause(1e-10)

            pos, array_masked = Helicopter1.return_q_view()
            a[:, pos - 1] += array_masked

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
            # Save Q - Matrix
        else:
            name = 'alpha_{}_epsilon_{}_gamma_{}_trails_{}_nb_actions_{}_model_{}'.format(
                settings['alpha'],
                settings['epsilon'],
                settings['gamma'],
                settings['trials'],
                settings['nb_actions'],
                settings['model'])
            # Saving the Neural Net Weights
            Helicopter1.ai.save_model(name=name)

# Save Settings
# Save History
