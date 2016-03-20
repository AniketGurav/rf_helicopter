### Overview of the Project

#### TODOs:

Updating README file:

* ADD VIDEO
* DISCUSS HOW THE GRIDWORLD (wind and track generation)
* ENHANCE PROJECT DESCRIPTION
* DISCUSS CHALLENGES WITH THE IMPLEMENTATION AND HOW THEY WERE OVERCOME

Upload all MODEL files:

* FINISHED DQN TESTING

Other Items:

* CLEAN CODE AND FINISH py.test

#### Project Objective:

Create a Reinforcement Learning algorithm to learn to play and complete the track, similar to the [Helicopter Game](http://www.helicoptergame.net/).

#### State Space (Partially Observable):

The reasoning behind why we chose a partially observable state space, a field of view, is that in real world situations it’s rare that the full state of the system can be provided to the agent or even determined. A real-life example is equivocal to a pilot in a plane where the pilot is equipped with a radar such that he can increase his field of view, in turn, enabling a greater oversight of his current situation. For this reasoning for our model and world, it was only provided with a small field of view – as shown by the grid space on the right-hand side of the helicopter in the figure below. Previous reports, Minh et al., 2015 and Minh et al., 2013, considered the observable whole state space and considered 4-10 frames as input into the Network. It is important to note here that this method attempts to embed knowledge into the Network, such that it can consider previous frames in addition to the current frame in order to determine Agents next action. A similar approach is used in our simulation were the model has access to the current and previous states for each training and prediction examples.

#### Models Implemented:

1. e-Greedy Q Learning (1)
2. e-Greedy Q-Learning with e-decay (2)
3. Deep Q-Network (*DQN*)(3)

#### DQN Notes:

The sequential nature of our problem means that it could potentially benefit from using a Recurrent Neural Network either as primary function approximator or as additional layers in the network. The implementation initially developed made use of similar/simpler architecture as proposed in previous work. This architecture that was implemented consisted of an Embedding Layer, which mapped the integer values (state) to a one-dimensional vector which were then passed into a Convolutional layer, followed by a Max-Pooling layer, lastly into a two Dense layers one with a Relu activation and the last a softmax function. The intention was to replace the Dense layers with Recurrent layers, however, throughout testing, it was found that stability and the time to experiment with varying factors meant that the decision to use the potentially enhanced version could not be achieved with the time frame. Lastly, our implementation also differs with previous was that it made use of a Softmax activation as the output – it was found that the model converged quicker and was also more stable during the training phase.

#### Track Generation:

For this project we developed a couple of scripts to create tracks, the tracks comprise of some obstacles that can either protrude from the ceiling of the track or upwards from the floor. To implement this, we first generate a random number of tuples which contain four attributes; width and height of the obstacles, starting location in the window and if it is on the ceiling or floor of the window. The set of tuples that are generated can then be made into windows, where a window will contain one obstacle. To create the track, a random number of windows are then selected and concatenated to make a track. At the window generation phase, the window is initially created with zeros and ones where ones are the obstacle. We then fill the zeros in with a function that to generate depending on the cells index a continuous value which can then be binned into seven distinct bins which correspond to different types of wind.

#### File Structure:

- rf_helicoper/
  - Model/ : Contains all the Scripts create tracks, generate plots and Agent model types
  - Results/ : Contains all the saved memories and plots generated
  - Tests/ : Started working on py.tests to ensure code quality.

#### Other Information:

This project differs in that for the University Course we were required to look at different cases to understand the impact of changing the values for each Case. The different Cases can be found in the ```Settings.py``` file, selecting the Model selection can be found in that file which will be used for both Training and Testing.

* Case 1: Create task that a Reinforcement Learning Alogorithm could solve. 
* Case 2: repeating the experiment in Case 1 with different gamma values
* Case 3: repeating the experiment in Case 1 with different learning rates
* Case 4: repeating the experiment in Case 1 with different policies
* Case 5: repeating the experiment in Case 1 with different state and reward functions

#### References:

* Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, et al. ‘Human-Level Control through Deep Reinforcement Learning’. Nature 518, no. 7540 (25 February 2015): 529–33. doi:10.1038/nature14236

* Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., Riedmiller, M., 2013. Playing Atari with Deep Reinforcement Learning. arXiv preprint arXiv:1312.5602.

