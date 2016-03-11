# Purpose: Main Function - Training Models
#   Info: Change the Parameters at the top of the scrip to change how the Agent interacts
#   Developed as part of the Software Agents Course at City University
#   Dev: Dan Dixey and Enrico Lopedoto
#   Updated: 1/3/2016

import logging
import os
import sys
from time import time
import json
import pylab

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

################ Model Settings
case = 'case_five'
settings_ = case_lookup[case]
iterations, settings = get_indicies(settings_)
file_name="Track_1.npy"

################ Plot Settings
plot_settings = dict(print_up_to=-1,
                     end_range=list(range(settings['trials'] - 0,
                                          settings['trials'] + 1)),
                     print_rate=1)

# Plotting Colors
colors = ['coral', 'green', 'red', 'cyan', 'magenta',
          'yellow', 'blue', 'white', 'fuchsia', 'orangered', 'steelblue']
          

HeliWorld = W.helicopter_world(file_name=file_name)
# file_name=None - Loads a Randomly Generated Track
Helicopter1 = helicopter(world=HeliWorld, settings=settings)

st = time()
time_metrics = []
results = dict(time_chart=[],
               final_location=[],
               best_test=[],
               q_plot=[],
               model_names=[],
               q_matrix=[],
               paths=[],
               returns=[])

t_array = []  # Storing Time to Complete
f_array = []  # Storing Final Locations
b_array = []  # Storing Full Control
a_array=[]
a = np.zeros(shape=(HeliWorld.track_height,
                    HeliWorld.track_width))
path = []

fig = plt.figure()
for value_iter in range(iterations):
    if case != 'case_one':
        rowz = 5
        colz = 2
        indexz = value_iter+1
        figsize=(15, 15)
    else:
        rowz = 1
        colz = 1
        indexz =1
        figsize=(15, 15)
    if value_iter > 0:
        settings = get_settings(dictionary=settings_,
                                ind=value_iter)
        HeliWorld = W.helicopter_world(file_name=file_name)
        Helicopter1 = helicopter(world=HeliWorld,
                                 settings=settings)
        a = np.zeros(shape=(HeliWorld.track_height,
                            HeliWorld.track_width))
        t_array = []  # Storing Time to Complete
        f_array = []  # Storing Final Locations
        b_array = []  # Storing Full Control
        a_array = []  #storing matrix value
        logging.info('Changing Values: {}'.format(settings_['change_values']))

    while HeliWorld.trials <= settings['trials']:
        # On the Last Trail give the Model full control
        if HeliWorld.trials == settings[
                'trials'] or HeliWorld.trials in plot_settings['end_range']:
            Helicopter1.ai.epsilon, settings['epsilon'] = 0, 0

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
                f_array.append(
                    [HeliWorld.trials, Helicopter1.current_location[0]])
                Helicopter1.reset()
                rate = (time() - st + 0.01) / HeliWorld.trials
                value = [HeliWorld.trials,
                         rate]
                t_array.append(value)
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
            a_array.append([HeliWorld.trials,a.sum()])
        HeliWorld.trials += 1

################################
        
    plt.subplot(rowz, colz, indexz)
    plt.title('Q Plot of Helicopter Path', fontsize=8)
    plt.xlabel('Track Length', fontsize=8)
    plt.ylabel('Track Width', fontsize=8)
    my_axis = plt.gca()
    my_axis.set_xlim(0, HeliWorld.track_width)
    my_axis.set_ylim(0, HeliWorld.track_height)
    im1 = plt.imshow(a,
                     cmap=plt.cm.jet,
                     interpolation='nearest')
    plt.colorbar(im1, fraction=0.01, pad=0.01)
    plt.show()
    
    name = 'model_{}_case_{}_iter_{}'.format(
        settings['model'],
        case.split('_')[1],
        value_iter)

    # Record Results
    results['time_chart'].append(t_array),
    results['final_location'].append(f_array)
    results['best_test'].append(b_array)
    results['q_plot'].append(a.tolist())
    results['model_names'].append(settings)
    results['returns'].append(a_array)

    et = time()

################################
xlim_val = int(settings['trials'])
nb_action = int(settings['nb_actions'])
n_items = len(results['best_test'])

## Save all results to a JSON file
#f = open(
#    os.path.join(
#        os.getcwd(),
#        'Results',
#        case,
#        'Model{}'.format(
#            settings['model']) +
#        '.json'),
#    'w').write(
#    json.dumps(results))
################################

model_plot = plotting_model()
model_plot.get_q_matrix(model_q=Helicopter1.ai.q,
                        nb_actions=settings['nb_actions'])
model_plot.plot_q_matrix('Q-Matrix - {}'.format(name))
q_data = model_plot.get_details()
results['q_matrix'].append(q_data)
plt.show()

################################

fig = plt.figure(figsize=figsize)
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

# For each set of results in dictionary
for i in range(n_items):
    x, y = [], []
    for each_item in results['best_test'][i]:
        x.append(each_item[0])
        y.append(each_item[1])
    # Plot Scatter
    plt.scatter(x=x,
                y=y,
                s=np.pi * (1 * 1) ** 2,
                c=colors[i])
plt.show()

################################

fig = plt.figure(figsize=figsize)
selection = choice(range(len(results['best_test'])))
plt.title('Final Q Matrix', fontsize=10)
plt.xlabel('Track Length', fontsize=8)
plt.ylabel('Track Width', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, HeliWorld.track_width)
my_axis.set_ylim(0, HeliWorld.track_height)
q_data = np.array(results['q_plot'][selection])
im1 = plt.imshow(q_data,
                 cmap=plt.cm.jet,
                 interpolation='nearest')
plt.colorbar(im1, fraction=0.01, pad=0.01)
plt.show()
    
################################
par=np.arange(0.1, 1.1, 0.1)

fig = plt.figure(figsize=figsize)
plt.title('Completion Chart - Time per Trial', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('LOG(Seconds Per Trial)', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, xlim_val)
# For each set of results in dictionary
for i in range(n_items):
    x, y = [], []
    for each_item in results['time_chart'][i]:
        x.append(each_item[0])
        y.append(each_item[1])
        # Plot Scatter
    plt.scatter(x=x,
                y=np.log(y),
                s=np.pi * (1 * 1) ** 2,
                c=colors[i],
                label=par[i])
                
plt.legend(title="Parameters")
plt.show()

################################
par=np.arange(0.1, 1.1, 0.1)

fig = plt.figure(figsize=figsize)
plt.title('Reward Chart - Tot Q Values per Trial', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('LOG(Q Values)', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, xlim_val)
# For each set of results in dictionary
for i in range(n_items):
    x, y = [], []
    for each_item in results['returns'][i]:
        x.append(each_item[0])
        y.append(each_item[1])
        # Plot Scatter
    plt.scatter(x=x,
                y=np.log(y),
                s=np.pi * (1 * 1) ** 2,
                c=colors[i],
                label=par[i])
    
plt.legend(title="Parameters")
plt.show()

################################

#fig = plt.figure(figsize=figsize)
#plt.title('Learning Chart - Averaged Trial Plot', fontsize=10)
#plt.xlabel('Trial Numbers', fontsize=8)
#plt.ylabel('End Location', fontsize=8)
#for i in range(n_items):
#    x, y = [], []
#    for each_item in results['final_location'][i]:
#        x.append(each_item[0])
#        y.append(each_item[1])
#    y = y
#    plt.plot(x, y, linewidth=0.5, c=colors[i])
#    
#title_text = '|| Case - {} | Number of Trials - {} | Model - {} | Number of Actions - {} ||\n\
#              || TRACK | Width - {} | Height - {} ||'.format(case,
#                                                             xlim_val,
#                                                             settings['model'],
#                                                             nb_action,
#                                                             HeliWorld.track_width,
#                                                             HeliWorld.track_height)
#fig.suptitle(title_text)
#plt.show()

################################

fig = plt.figure(figsize=figsize)
plt.hist(results['q_matrix'][0]['data'], bins=50)
plt.title("Q-Value Distribution - Min={} to Max={}".format(
    results['q_matrix'][0]['min'], results['q_matrix'][0]['max']))
plt.xlabel("Value")
plt.ylabel("Frequency")
fig.suptitle('Q Matrix')
plt.show()      