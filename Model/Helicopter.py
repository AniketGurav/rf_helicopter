# Purpose: Class that Controls the Movement of the Agent
#
#   Info: Main Function for linking classes togeather
#
#   Developed as part of the Software Agents Course at City University
#
#   Dev: Dan Dixey and Enrico Lopedoto
#
#
import logging
import os
import sys
import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from Agent import agent_controls
import Q_Learning_Agent as Q


# Logging Controls Level of Printing
logging.basicConfig(format='[%(asctime)s] : [%(levelname)s] : [%(message)s]',
                    level=logging.INFO)


class helicopter(agent_controls):

    def __init__(self, world, settings):
        agent_controls.__init__(self)
        self.ai = None
        self.model_version = settings['model']
        self.world = world
        self.settings = settings
        if self.model_version == 1:
            self.ai = Q.Q_Learning_Algorithm(settings=settings)
        elif self.model_version == 2:
            self.ai = Q.Q_Learning_Epsilon_Decay(settings=settings)
        elif self.model_version == 3:
            self.ai = Q.Q_Neural_Network(settings=settings,
                                         track_height=self.world.track_height)

        # Agent Metrics
        self.crashed = 0
        self.completed = 0
        # Storing States
        self.lastState = None
        self.current_state = None
        # Storing Locations
        self.origin = (world.st_x, world.st_y)
        self.current_location = self.origin
        self.previous_location = None
        # Storing Actions
        self.lastAction = None
        # Agents Info
        self.final_location = []
        self.q_matrix = []  # Q-Matrix state(p) vs state(c) - Q-Value
        self.r_matrix = []  # R_Matrix state(c) vs Action  - Reward
        self.trial_n = 1
        # Recording States
        self.state_record = []
        # Reward Functions
        self.reward_completed = settings['completed']
        self.reward_crashed = settings['crashed']
        self.reward_no_obstacle = settings['open']
        self.reward_sum = 0
        self.prev_reward = None
        self.new_state = None

    def update(self):
        # Get the Current State
        location = self.current_location
        world_val = self.world.check_location(location[0],
                                              location[1])
        state = self.find_states(self.current_location)

        # Record State
        self.state_record.append(state)

        # Is Current State Obstacle?
        if world_val == -1:
            self.crashed += 1
            self.reward_sum += self.reward_crashed
            self.prev_reward = self.reward_crashed
            if self.model_version == 3:  # Neural Network
                self.ai.update_train(p_state=self.lastState,
                                     action=self.lastAction,
                                     p_reward=self.reward_no_obstacle,
                                     new_state=state,
                                     terminal=[self.reward_completed,
                                               self.reward_crashed])

            if self.lastState is not None and self.model_version != 3:
                self.ai.learn(
                    self.lastState,
                    self.lastAction,
                    self.reward_crashed,
                    state)
            self.final_location.append([self.current_location[0],
                                        self.trial_n,
                                        self.current_location[1],
                                        self.reward_sum])
            self.r_matrix.append([self.lastState,
                                  self.lastAction,
                                  self.reward_crashed])
            self.q_matrix.append([self.lastState,
                                  state,
                                  self.reward_crashed])
            self.trial_n += 1
            # Agent Crashed - Reset the world
            return False

        # Is the Current State on the Finish Line?
        if world_val == 10:
            logging.debug("Helicopter Completed Course")
            self.completed += 1
            self.reward_sum += self.reward_completed
            self.prev_reward = self.reward_completed
            if self.model_version == 4:  # Neural Network
                self.ai.update_train(p_state=self.lastState,
                                     action=self.lastAction,
                                     p_reward=self.reward_no_obstacle,
                                     new_state=state,
                                     terminal=[self.reward_completed,
                                               self.reward_crashed])

            if self.lastState is not None and self.model_version != 3:
                self.ai.learn(self.lastState,
                              self.lastAction,
                              self.reward_completed,
                              state)
            self.final_location.append([self.current_location[0],
                                        self.trial_n,
                                        self.current_location[1],
                                        self.reward_sum])
            self.r_matrix.append([self.lastState,
                                  self.lastAction,
                                  self.reward_completed])
            self.trial_n += 1
            # Agent Completed Course - Reset the world
            return False

        # Is the Current in the Open - Continue Journey
        self.reward_sum += self.reward_no_obstacle
        self.prev_reward = self.reward_no_obstacle

        if self.lastState is not None and self.model_version != 3:
            self.ai.learn(self.lastState,
                          self.lastAction,
                          self.reward_no_obstacle,
                          state)

        # Select an Action
        if self.model_version < 4:
            action = self.ai.choose_Action(state)
        else:
            action = self.ai.choose_Action(state=state,
                                           pstate=self.lastState,
                                           paction=self.lastAction,
                                           preward=self.reward_no_obstacle)

        self.r_matrix.append([self.lastState,
                              self.lastAction,
                              self.reward_no_obstacle])
        self.q_matrix.append([self.lastState,
                              state,
                              self.reward_no_obstacle])

        self.lastState = state
        self.lastAction = action

        # Move Depending on the Wind at the current location
        self.current_location = self.action_wind(world_val,
                                                 self.current_location)
        if self.current_location is None:
            return False

        # Move Depending on the Action from Q-Learning
        self.current_location = self.action_move(action,
                                                 self.current_location)
        self.new_state = state
        if self.model_version == 3:  # Neural Network
            self.ai.update_train(p_state=self.lastState,
                                 action=self.lastAction,
                                 p_reward=self.reward_no_obstacle,
                                 new_state=state,
                                 terminal=[self.reward_completed,
                                           self.reward_crashed])

        return True

    def reset(self):
        self.current_location = self.origin
        self.previous_location = None
        self.lastAction = None
        self.lastState = None
        self.current_state = None
        self.reward_sum = 0

    def find_states(self, location):
        x, y = location[0], location[1]
        state_space = list()
        # Increase from 1 to 0
        for i in range(0, 3):
            for j in range(-2, 3):
                value = self.world.check_location(x=x + i,
                                                  y=y + j)
                state_space.append(value)
        # Add the current height into the state space.
        state_space.append(y)
        return tuple(state_space)

    def return_q_view(self):
        qw_mat = self.model_view()
        start = int(self.current_location[1])
        array1 = np.zeros(shape=(1, self.world.track_height + 3))
        array3 = np.array(qw_mat)
        array2 = np.ma.masked_array(array3, mask=[5])

        # Dealing with Edge Plotting
        lower = max(start - 2, 0)
        upper = min(start + 3, self.world.track_height + 1)
        array1[0, lower:upper] = array2[:upper - lower]

        return min(self.current_location[0], self.world.track_width), \
            array1[0, :self.world.track_height]

    def model_view(self):
        view_current = self.q_matrix[- 1][1]
        qw_mat = []
        if self.model_version < 3:
            for i in range(self.settings['nb_actions']):
                key = (view_current, i + 1)
                if key not in list(self.ai.q.keys()):
                    qw_mat.append(0)
                else:
                    qw_mat.append(self.ai.q[key])
        else:
            state = np.concatenate(
                (list(
                    self.lastState), [
                    self.lastAction], [
                    self.ai.reward_change[
                        self.prev_reward]], list(
                        self.new_state))) + 1
            state = np.asarray(state).reshape(1, self.ai.input_dim)
            qw_mat = self.ai.model.predict(state, batch_size=1)
        return qw_mat
