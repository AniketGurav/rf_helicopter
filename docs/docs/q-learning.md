#### Models Implemented:

1. e-Greedy Q Learning (1)
2. e-Greedy Q-Learning with e-decay (2)
3. Deep Q-Network (*DQN*)(3)

#### Overview of Q-Learning

All the models discussed in the coursework will use an online update policy strategy – an epsilon-greedy which intends to ensure adequate exploration of all the state space. Rummery and Niranjan (Rmmery et al., 1994) [9] provides an example of setting in which similar procedure are adopted. To exploit differences in the result of this self-driving helicopter simulation, we will compare three methods in which a different policy has been applied to select the best actions.

Q-learning uses temporal differences to estimate the value of Q*(s,a). In Q-learning, the agent maintains a table of Q[S,A], where S is the set of states and A is the set of actions. Q[s,a] represents its current estimate of Q*(s,a).

The Q-Learning variables are represented below:

*   Q(s, a) = Q value of a given state and action
*   a = Action
*   s = State
*   r = Reward
*   R = Maximum reward for action a

Where Q-Learning can be represented by:

    Q[s,a] ←Q[s,a] + α(r+ γmaxa' Q[s',a'] - Q[s,a])

#### Motivation for DQN

Last year, Deep Q Networks (DQN) were brought to the attention of many researchers when Deepmind released a paper demonstrating the network's capability at playing Atari games.

The research featured in the Nature publication and showed that their implementation had overcome the issues that had typically been challenged when using a Neural Network as a function approximation for the Q values. Summarized in the table below, from the paper Playing Atari with Deep Reinforcement Learning (Deepmind, 2015) [16], the issue is discussed as well as the techniques that have been used to overcome these problems.

| Issues                                      | Techniques                  |
|---------------------------------------------|-----------------------------|
| Stability Issues                            | Reward Clipping             |
| Distribution of the data can change quickly | Error Clipping (Truncation) |

The issues outlined in the table were implemented and were shown to have a real impact on the capability of the approximation. The analysis indicated that normalizing the range of the reward to a finite range helped to support the issue of dealing with large Q-values and their respective gradients – one negative of this approach was that the model may find it harder to differentiate the difference between small and large rewards due to the normalization. The second technique that was introduced was error clipping into the model – this is a frequently used method to deal with the potential of exploding gradients. The pseudocode below is a high-level description of the methods that were implemented in the Deepmind paper and in our model.

#### Implementing the DQN

The sequential nature of our problem means that it could potentially benefit from using a Recurrent Neural Network either as primary function approximator or as additional layers in the network, as in Mnih (Mnih et al., 2015) [15]. The original paper described that a Convolutional Neural Network (CNN) was used to “watch” the replays of the game. In our implementation of the Deep Q-Network, we used a CNN that outputs to a Long Short-Term Memory (LSTM) layer and then finally into a linear output layer providing the Q-Values from the model. The key distinction to the original paper was that an LSTM layer was used where it has been demonstrated in many papers previously that an RNN is accomplished at capturing temporal patterns in sequential data. Since the goal of Q-Learning is to learn good policies for sequential decision making it, therefore, seemed appropriate to include this layer type.

The implementation initially developed made use of similar/simpler architecture to that proposed in previous work. This architecture that was implemented consisted of an Embedding Layer, which mapped the integer values (state) to a one-dimensional vector which were then passed into a Convolutional layer, followed by a Max-Pooling layer and lastly into two Dense layers (one with a Relu activation and the last a Softmax function). The intention was to replace the Dense layers with Recurrent layers, however, throughout testing, it was found that stability and the time to experiment with varying factors meant that the desire to move to the potentially enhanced version could not be achieved within the time frame. One feature that was implemented was that the model could record and store instances of the transitions and experience replay. This idea was used in one the most successful use cases of Neural Networks in Reinforcement Learning; this model was called TD-Gammon.

The model was developed in Keras, a Deep Learning library for Python, the primary reason for this is that it provides a very easy layer of abstraction on top of Theano or Tensorflow.

The diagram below captures the architecture used for developing the model.

*placeholder: field of view diagram*

#### Expectations

For both all of the Q-Learning variants it is expected that:

1.  the model will probably go back in forth between preferring different actions over others during the initialisation
    1.  finding an appropriate learning rate and Gammma value for will be crucial
    2.  ensuring that the terminal state can be reached otherwise not enough learning can be stimulated
2.  DQN learning models will be sufficiently more difficult to training
    1. tuning of both the model and q-learning parameters will be critical to ensure the model converges
