# Purpose: Contains Util Variables and Functions
#
#   Info: Stuff that belongs not in model
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#
import numpy as np


titles = ['Q-Learning Algorithm',
          'Q-Learning Algorithm with Learning Rate Decay',
          'Q-Learning Algorithm with Temporal Difference',
          'Q-Learning Algorithm with Neural Network']


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


case_lookup = dict(case_two=['gamma'],
                   case_three=['alpha'],
                   case_four=['epsilon'],
                   case_five=['Completed', 'Crashed', 'Open'])


def get_string(dictionary):
    s = ''
    for each_value in dictionary['change_values']:
        s = s + ' {}: {}'.format(each_value, dictionary[each_value])
    return s
