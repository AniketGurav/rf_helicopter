# Overview

**Objective**: 

Create a Reinforcement Learning algorithm to learn to play and complete the track, similar to the [Helicopter Game](http://www.helicoptergame.net/).

The state is defined by the Field of View that the Helicopter is able to percieve in front of itself. In the current configuration the is a 3x5 grid, in addition to this the model also gets given the height from the floor of the grid world.

Implementation details for the Q-Learning Algorithm have been taken from the [Playing Atari with Deep Reinforcement Learning.](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). Notable implemented features from DeepMinds work included:

* Experience Replay
* Epsilon Decay during Training
* A version of Reward Clipping/ Scaling

**Info**: 

Descriptions added as headers for each file. Main files to run and plot models are: main.py and plot_results.py

**Models**: 

1. e-Greedy Q Learning (1)
2. e-Greedy Q-Learning with e-decay (2)
3. Deep Q-Network (*DQN*)(3)

**Files**:

* **Train.py** - Train a Model (Saves Memory and Metrics to JSON and Pickle files)
* **Test.py** - Test the Model on an Unseen Track and Plots Results
*  **Plot_results.py** - Plot results from the Models generated from the Train file

**File Structure**:

- rf_helicoper/
  - Model/ : Contains all the Scripts create tracks, generate plots and Agent model types
  - Results/ : Contains all the saved memories and plots generated
  - Tests/ : Started working on integration tests (WIP)

**Other Information**:

The different cases can be found in the ```Settings.py``` file.

* Case 1: Create task that a Reinforcement Learning Alogorithm could solve. 
* Case 2: repeating the experiment in Case 1 with different gamma values
* Case 3: repeating the experiment in Case 1 with different learning rates
* Case 4: repeating the experiment in Case 1 with different policies
* Case 5: repeating the experiment in Case 1 with different state and reward functions


