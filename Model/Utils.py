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


# Model Labels
titles = ['Q-Learning Algorithm',
          'Q-Learning Algorithm with Learning Rate Decay',
          'Q-Learning Algorithm with Neural Network']


def moving_average(interval, window_size):
    """
    Simple Moving Average Function

    :param interval: np.array
    :param window_size: int
    :return: np.array
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


case_lookup = dict(case_two=['gamma'],
                   case_three=['alpha'],
                   case_four=['epsilon'],
                   case_five=['Completed', 'Crashed', 'Open'])


def get_string(data):
    """
    Get Key and Label formatted correctly

    :param data: dict
    :return: str
    """
    s = ''
    for each_value in data['change_values']:
        s = s + ' {}: {}'.format(each_value, data[each_value])
    return s
