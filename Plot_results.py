# Purpose: Main Plotting Results Script
#
#   Info: Plotting the Results by Case and Model (1, 2 or 3)
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#   Updated: 5/3/2016
#
import json
import logging
import os
from random import choice

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

from Model import World as W
from Model.Utils import moving_average

plt.style.use('ggplot')


# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.DEBUG)


case_name = 'case_four'
model = str(3)
directory = os.path.join(os.getcwd(), 'Results',
                         case_name)


data = json.loads(open(directory + '/Model{}.json'.format(model), 'r').read())
HeliWorld = W.helicopter_world(file_name="Track_1.npy")
xlim_val = int(data['model_names'][0]['trials'])
nb_action = int(data['model_names'][0]['nb_actions'])
n_items = len(data['best_test'])

# Plotting Colors
colors = ['coral', 'green', 'red', 'cyan', 'magenta',
          'yellow', 'blue', 'white', 'fuchsia', 'orangered', 'steelblue']

fig = plt.figure()
fig.canvas.draw()
plt.subplot(2, 2, 1)
plt.title('Post Training Path - Epsilon = 0',
          fontsize=10)
plt.xlabel('Track Length', fontsize=8)
plt.ylabel('Track Width', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, HeliWorld.track_width)
my_axis.set_ylim(0, HeliWorld.track_height)
im1 = plt.imshow(HeliWorld.track,
                 cmap=plt.get_cmap('gray'),
                 interpolation='nearest',
                 vmin=-1,
                 vmax=8)
plt.colorbar(im1, fraction=0.01, pad=0.01)

# For each set of results in dictionary
for i in range(n_items):
    x, y = [], []
    for each_item in data['best_test'][i]:
        x.append(each_item[0])
        y.append(each_item[1])
    # Plot Scatter
    plt.scatter(x=x,
                y=y,
                s=np.pi * (1 * 1) ** 2,
                c=colors[i])

logging.info('Plotting the Q-Matrix')
plt.subplot(2, 2, 2)
selection = choice(range(len(data['best_test'])))
plt.title('Final Q Matrix', fontsize=10)
plt.xlabel('Track Length', fontsize=8)
plt.ylabel('Track Width', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, HeliWorld.track_width)
my_axis.set_ylim(0, HeliWorld.track_height)
q_data = np.array(data['q_plot'][selection])
im2 = plt.imshow(normalize(q_data))
plt.colorbar(im2, fraction=0.01, pad=0.01)

logging.info('Completion Chart - Time per Trial')
plt.subplot(2, 2, 3)
plt.title('Completion Chart - Time per Trial', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('LOG(Seconds Per Trial)', fontsize=8)
my_axis = plt.gca()
my_axis.set_xlim(0, xlim_val)

# For each set of results in dictionary
for i in range(n_items):
    x, y = [], []
    for each_item in data['time_chart'][i]:
        x.append(each_item[0])
        y.append(each_item[1])
    # Plot Scatter
    plt.scatter(x=x,
                y=np.log(y),
                s=np.pi * (1 * 1) ** 2,
                c=colors[i])

plt.subplot(2, 2, 4)
plt.title('Learning Chart - Averaged Trial Plot', fontsize=10)
plt.xlabel('Trial Numbers', fontsize=8)
plt.ylabel('End Location', fontsize=8)

# For each set of results in dictionary
for i in range(n_items):
    x, y = [], []
    for each_item in data['final_location'][i]:
        x.append(each_item[0])
        y.append(each_item[1])
    y = moving_average(y, 60)
    plt.plot(x, y, linewidth=1, c=colors[i])

logging.info('Plotting Figure Label')
title_text = '|| Case - {} | Number of Trials - {} | Model - {} | Number of Actions - {} ||\n\
              || TRACK | Width - {} | Height - {} ||'.format(case_name,
                                                             xlim_val,
                                                             model,
                                                             nb_action,
                                                             HeliWorld.track_width,
                                                             HeliWorld.track_height)
fig.suptitle(title_text)
logging.info('Saved Figure of the Plot')
fig.savefig(directory + '/Plot/Final_Plot_{}.png'.format(model))


