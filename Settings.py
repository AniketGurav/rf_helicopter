# Purpose: Script containing Settings for the Model
#
#   Info: Change the Parameters at the top of the scrip to change how the Agent interacts
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#   Updated: 4/3/2016
#
import numpy as np

# Case 1 - Default Evaluation
case_one = dict(trials=200,
                completed=500,
                crashed=-100,
                open=5,
                alpha=0.65,
                epsilon=0.75,
                gamma=0.7,
                nb_actions=5,
                model=1,
                epsilon_decay=0.9,
                epsilon_action=6000,
                lambda_td=0.5,
                change_values=[])

# Case 2 - Change Gamma values
case_two = dict(trials=200,
                completed=500,
                crashed=-100,
                open=5,
                alpha=0.65,
                epsilon=0.75,
                gamma=np.arange(0.1, 1.1, 0.1),
                nb_actions=5,
                model=1,
                epsilon_decay=0.9,
                epsilon_action=6000,
                lambda_td=0.5,
                change_values=['gamma'])

# Case 3 - Change Learning Rates
case_three = dict(trials=200,
                  completed=500,
                  crashed=-100,
                  open=5,
                  alpha=np.arange(0.1, 1.1, 0.1),
                  epsilon=0.75,
                  gamma=0.7,
                  nb_actions=5,
                  model=1,
                  epsilon_decay=0.9,
                  epsilon_action=6000,
                  lambda_td=0.5,
                  change_values=['alpha'])

# Case 4 - different policies (epsilon)
case_four = dict(trials=200,
                 completed=500,
                 crashed=-100,
                 open=5,
                 alpha=0.65,
                 epsilon=np.arange(0.1, 1.1, 0.1),
                 gamma=0.7,
                 nb_actions=5,
                 model=1,
                 epsilon_decay=0.9,
                 epsilon_action=6000,
                 lambda_td=0.5,
                 change_values=['epsilon'])

# Case 5 - different Reward functions
case_five = dict(trials=200,
                 completed=np.arange(10, 500, 50),
                 crashed=np.arange(-10, -110, -10),
                 open=np.arange(0, 10),
                 alpha=0.65,
                 epsilon=0.75,
                 gamma=0.7,
                 nb_actions=5,
                 model=1,
                 epsilon_decay=0.9,
                 epsilon_action=6000,
                 lambda_td=0.5,
                 change_values=['completed',
                                'crashed',
                                'open'])

# Case Dictionary
case_lookup = dict(case_one=case_one,
                   case_two=case_two,
                   case_three=case_three,
                   case_four=case_four,
                   case_five=case_five)

# Model Colors
model_color = ["red",     # Q-Learning (e-greedy)
               "blue",    # Q-Learning (e-greedy with epsilon decay)
               "yellow"]  # Deep Q-Learning (DQN

# Get Indicies count


def get_indicies(dictionary, ind=0):
    if len(dictionary['change_values']) > 0:
        return 10, get_settings(dictionary)
    else:
        return 1, dictionary

# Get New Dictionary values


def get_settings(dictionary=None, ind=0):
    new_dict = dictionary.copy()
    for each_value in dictionary['change_values']:
        new_dict[each_value] = dictionary[each_value][ind]
    return new_dict
