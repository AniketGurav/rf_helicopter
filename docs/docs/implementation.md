## Track Generation:

For this project we developed a couple of scripts to create tracks, the tracks comprise of some obstacles that can either protrude from the ceiling of the track or upwards from the floor. To implement this, we first generate a random number of tuples which contain four attributes; width and height of the obstacles, starting location in the window and if it is on the ceiling or floor of the window. The set of tuples that are generated can then be made into windows, where a window will contain one obstacle.

To create the track, a random number of windows are then selected, trimmed and concatenated to make a track. At the window generation phase, the window is initially created with zeros and ones where ones are the obstacle. We then fill the zeros in with a function (```x**2 + 2 * y**2```) to generate a continuous value, which are then be binned into seven distinct bins which correspond to different actions of the wind.

There are two files that can be used to generate tracks - with and without wind is possible.

    /Model/Build_tracks.py
    /Model/Build_tracks_wind.py

These python scripts simply calling the classes contained within the following files to get the respective tracks.

    /Model/Generate_obstacles.py
    /Model/Wind_Generation.py

## Agent

    /Model/Agent.py

This file contains the mapping for action to location update with respect to the world. There are two functions:

1.  Action Controller : this controls the behaviour of the Agent that it has the ability to change
2.  Wind Controller : dependinng on the location (x, y) that the Agent is currently at dictates the resulting action of the Agent. The Q-values that are calculate are independent of this Controller.

## Q-Learning Classes

    /Model/Q_Learning_Agent.py

Within this file there are 3 classes - one per type of model. At the heart of all three classes is there Q-Learning function.

    def learnQ(self, state, action, reward, value):

        old_value = self.q.get((state, action), None)

        if old_value is None:
            self.q[(state, action)] = reward

        else:
            self.q[(state, action)] = old_value + \
                self.alpha * (value - old_value)

The three models are:

    1. e-Greedy Q Learning
    2. e-Greedy Q-Learning with epsilon decay
    3. Deep Q-Network (*DQN*)

### Models 1-2

Both models make use of a python dictionary in order to map State-Action pairs to rewards. By using this method as opposed to a pre-computed table is that there was no startup cost associated with calculating them inadvance of training the model.

The primary disadvantage is that if the model was used in a predictive state and the Agent had not seen a particular pair before then it's performance with regards to the optimal policy be terrible. It would therefore be deemed that the model is generalising poorly - it would be hoped that in the finite state space that the model is learning that this would not be the case.

### Model 3

Model 3 uses a Neural Network to Approximate the

## Challenges and Issues Resolved

TODO

