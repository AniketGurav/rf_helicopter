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
from time import time, sleep
import json

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from Model.Helicopter import helicopter
from Model import World as W
from Model.Plotting import plotting_model
from Settings import *
import matplotlib
import matplotlib.pyplot as plt
from random import choice
matplotlib.style.use('ggplot')


# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.DEBUG)


logging.info("Setting Parameters:")
# Model Settings
case = 'case_one'
settings_ = case_lookup[case]
iterations, settings = get_indicies(settings_)

# Plot Settings
plot_settings = dict(print_up_to=-1,
                     end_range=list(range(30,
                                          60)),
                     print_rate=5)

logging.info("Load Helicopter and World")
HeliWorld = W.helicopter_world(file_name="Track_1.npy")
# file_name=None - Loads a Randomly Generated Track
Helicopter1 = helicopter(world=HeliWorld,
                         settings=settings)
# Load Helicopter1 Weights (model=3)
if settings['model'] == 3:
    logging.info('Loading Saved Model')
    value_iter, model = 0, settings['model']
    name = 'model_{}_case_{}_iter_{}'.format(
        settings['model'],
        case.split('_')[1],
        value_iter)
    Helicopter1.ai.load_model(name=name)
    Helicopter1.ai.update_rate = 10000000
    logging.info('Loaded Saved Model')

settings['trials'] = 60
Helicopter1.ai.epsilon = 0

a = np.zeros(shape=(HeliWorld.track_height,
                    HeliWorld.track_width))

logging.info("Starting the Learning Process")
st = time()
time_metrics = []
b_array = []

results = dict(paths=[])
path = []

logging.info('Dealing with Case: {}'.format(case))
for value_iter in range(iterations):
    if value_iter > 0:
        settings = get_settings(dictionary=settings_,
                                ind=value_iter)
        HeliWorld = W.helicopter_world(file_name="Track_1.npy")
        Helicopter1 = helicopter(world=HeliWorld,
                                 settings=settings)
        a = np.zeros(shape=(HeliWorld.track_height,
                            HeliWorld.track_width))
        logging.info('Changing Values: {}'.format(settings_['change_values']))

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
            if HeliWorld.trials == settings['trials']:
                b_array.append(Helicopter1.current_location)
            if not output:

                Helicopter1.reset()
                rate = (time() - st + 0.01) / HeliWorld.trials
                value = [HeliWorld.trials,
                         rate]
                if HeliWorld.trials <= plot_settings[
                        'print_up_to'] or HeliWorld.trials in plot_settings['end_range']:
                    results['paths'].append(path)
                    path = []
                break

            if HeliWorld.trials <= plot_settings[
                    'print_up_to'] or HeliWorld.trials in plot_settings['end_range']:
                # Primary Title
                rate = (time() - st + 0.01) / HeliWorld.trials
                value = [HeliWorld.trials,
                         rate]
                path.append(Helicopter1.current_location)

            pos, array_masked = Helicopter1.return_q_view()
            a[:, pos - 1] += array_masked

        logging.debug('Starting next iteration')
        HeliWorld.trials += 1

    et = time()
    logging.info(
        "Time Taken: {} seconds for Iteration {}".format(
            et - st, value_iter + 1))

fig = plt.figure()
plt.title('Real-time Plot of Helicopter Path', fontsize=10)
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
# Plotting Colors
colors = ['black', 'green', 'red', 'cyan', 'magenta',
          'yellow', 'blue', 'white', 'fuchsia', 'orangered', 'steelblue']

for val, data in enumerate(results['paths']):
    x, y = [], []
    for step in data:
        x.append(step[0])
        y.append(step[1])
    plt.scatter(x,
                y,
                s=np.pi * (1 * (1.5))**2,
                c=choice(colors))
    plt.pause(0.5)
    sleep(0.5)

fig1 = plt.figure()
plt.title('Q Plot of Helicopter Path', fontsize=10)
plt.xlabel('Track Length', fontsize=8)
plt.ylabel('Track Width', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, HeliWorld.track_width)
my_axis.set_ylim(0, HeliWorld.track_height)
im1 = plt.imshow(a,
                 cmap=plt.cm.jet,
                 interpolation='nearest')
plt.colorbar(im1, fraction=0.01, pad=0.01)
