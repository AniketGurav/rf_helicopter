#### Helicopter Game

The purpose of this analysis is to provide the reader a concrete example of a classical Reinforcement Learning application.

The intention is to create a Reinforcement Learning algorithm to learn to play and complete the track, similar to the [Helicopter Game](http://www.helicoptergame.net/). The domain of implementation will be the contest of a randomly generated environment according to a particular function, and the learning task will provide the Agent a set of information enabling him to survive in the current environment for a longer duration of time.

The environment will provide to the Agent the possibility to move in five possible actions (directions), and the survival function will be defined as that function allowing the Agent to avoid obstacles generated in the environment.

Proposed algorithms for this task are all online policy iterations models with e-greedy updates policies: Q-Learning, Q-Learning with Epsilon Decay, and Deep Q-Learning Networks (DQN). A comparison of different parameters will be performed and evaluated.

#### Problem Representation

The helicopter task can be summarized as a survival task - the longer it travels along the path, the better its performance. However, the maximum reward is gained only by reaching the end of the track. The States following are represented by the different gradient colours in the part of the track which is not an obstacle, and the wind function guarantees a state which ranges from 0 to 6 according to which the helicopter is taking an action. Obstacles are initially set to be -10 and final reward to 100. The representation of the best route is the one by which the helicopter navigates the wind effectively and reaches the end of the track without any interruptions.

There are a number of possible actions it can take are either up, down, or continue laterally; different index increments dictate different actions. This index can range from 1 to 5, where an index of 1 would attempt to move the helicopter down by two, and an index of 5 would move it up by two. The new state is also moved horizontally by one enabling it to move toward the end of the track. Once the Agent’s actions have been applied the wind value an additional action is then applied to the helicopter. An example of a possible action is to increase the Agents location by 2 in the vertical plane.

#### Domain

The domain of application is state space environment in which our entity, the “helicopter” will try to self-drive to live as long as possible in the randomly generated path. The environment is created as a matrix of variable size in both its coordinates of length and width. The entity's task is to complete the matrix starting from the left side and tries to travel successfully to the right side without interruption represented by the randomly generated obstacles.

We refer to this task as the example of a helicopter that has to maintain certain minimum and maximum quotes corresponding to the generate obstacles. To add small complexities, a wind model has been produced and also the Agent will be only given a partially observable state. To deal with the long-term deficiency of not receiving any reward during flight, the model receives a small reward as it navigates to the end of the window.

#### State Space (Partially Observable):

The reasoning behind why we chose a partially observable state space, a field of view, is that in real world situations it’s rare that the full state of the system can be provided to the agent or even determined. A real-life example is equivocal to a pilot in a plane where the pilot is equipped with a radar such that he can increase his field of view, in turn, enabling a greater oversight of his current situation. For this reasoning for our model and world, it was only provided with a small field of view – as shown by the grid space on the right-hand side of the helicopter in the diagram below.

Previous reports, Minh et al., 2015 and Minh et al., 2013, considered the observable whole state space and used 4-10 frames as input into the Network. It is important to note here that this method attempts to embed knowledge into the Network, such that it can consider previous frames in addition to the current frame in order to determine Agents next action. A similar approach is used in our simulation were the model has access to the current and previous state for each training and prediction.

*placeholder: field of view diagram*

#### Models Implemented:

1. e-Greedy Q Learning
2. e-Greedy Q-Learning with epsilon decay
3. Deep Q-Network (*DQN*)

#### Other Remarks:

This project differs was prepared for a University Course, which required the investigation to look at different Cases to understand the impact of changing the values for each parameter. The different Cases and selected can be found in the ```Settings.py``` file, selecting the Model selection can also be found in that file - the file is used in both the Training and Testing scripts.

* Case 1: Create task that a Reinforcement Learning Alogorithm could solve.
* Case 2: repeating the experiment in Case 1 with different gamma values
* Case 3: repeating the experiment in Case 1 with different learning rates
* Case 4: repeating the experiment in Case 1 with different policies
* Case 5: repeating the experiment in Case 1 with different state and reward functions
