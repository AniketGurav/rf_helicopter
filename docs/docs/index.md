## Project Objective:

The purpose of this analysis is to provide the reader a concrete example of a classical Reinforcement Learning application.

The intention is to create a Reinforcement Learning algorithm to learn to play and complete the track, similar to the [Helicopter Game](http://www.helicoptergame.net/). The domain of implementation will be the contest of a randomly generated environment according to a particular function, and the learning task will provide the Agent a set of information enabling him to survive in the current environment for a longer duration of time.

The environment will provide to the Agent the possibility to move in five possible actions (directions), and the survival function will be defined as that function allowing the Agent to avoid obstacles generated in the environment.

Proposed algorithms for this task are all online policy iterations models with e-greedy updates policies: Q-Learning, Q-Learning with Epsilon Decay, and Deep Q-Learning Networks (DQN). A comparison of different parameters will be performed and evaluated.

## City University

This project was completed as part of a University course work for which a **First Class** award was received.

!!! note
    Project Structure:

- `rf_helicoper/`
  - `/Model` : Contains all the Scripts create tracks, generate plots and Agent model types
  - `/Results` : Contains all the saved memories and plots generated
  - `/Tests` : Started working on py.tests to ensure code quality
  - `/docs` : MKDocs documentation
