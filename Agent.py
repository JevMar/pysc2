from pysc2.agents import base_agent
from pysc2.lib import features
from pysc2.lib import actions

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

from collections import deque
import pandas as pd
import numpy as np
import os.path
import random
import re

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_GHOST = actions.FUNCTIONS.Train_Ghost_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_ARMY_CAP_USED = 3
_ARMY_CAP = 4
_ARMY_SUPPLY = 5
_ARMY_COUNT = 8
_SPENT_MINERALS = 11
_SPENT_VESPENE = 12
_KILLED_UNITS = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_BUILD_REAPER = 'buildreaper'
ACTION_BUILD_MARAUDER = 'buildmarauder'
ACTION_BUILD_GHOST = 'buildghost'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_MARINE,
    ACTION_BUILD_REAPER,
    ACTION_BUILD_MARAUDER,
    ACTION_BUILD_GHOST,
    ACTION_ATTACK
]


class DQN:
    def __init__(self, states, actions):
        self.states = np.array(states)
        self.actions = np.array(actions)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005

        self.model = self.create_model()

        self.tau = .125  # need?

    def create_model(self):
        model = Sequential()
        model.add(Dense(7, input_dim=1, activation="relu"))  # why 7?
        model.add(Dense(self.actions.shape[0]))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy', 'mae'])
        return model

    def act(self, state):
        if np.random.random() < self.epsilon:
            rnd_choice = np.random.choice(self.actions, size=1, replace=False)[0]
            return rnd_choice
        else:
            pred_choice = np.argmax(self.model.predict(state)[0])
            return pred_choice

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size):
        print('Replay - memory:{0}; batch_size:{1}'.format(self.memory, batch_size))
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if len(re.findall("[0-9]", str(action))) > 0:
                action = action
            else:
                action = smart_actions.index(str(action))

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class ddqnAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ddqnAgent, self).__init__()

        self.reward = 0
        self.episode = 1
        self.steps = 0
        self.ddqn = DQN([0, 0, 0], smart_actions)
        self.batch_size = 50

        self.previous_action = None
        self.previous_state = None
        self.smart_action = None

        self.previous_killed = 0
        self.killed_score = 0

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

    def reset(self):
        self.reward = 0

        if (self.episode > 9) and (self.episode % 10 == 0):
            self.ddqn.replay(self.batch_size)

    def step(self, obs):
        super(ddqnAgent, self).step(obs)

        unit_type = obs.observation['screen'][_UNIT_TYPE]
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()

        killed_units = obs.observation['score_cumulative'][_KILLED_UNITS]

        if obs.first():
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        if obs.last():
            # reward = obs.reward
            self.episode += 1

            if killed_units > self.previous_killed < 1:
                self.killed_score += 0.25

            print('reward,killed,army,score;{0};{1};{2};{3}'.format(self.reward,
                                                                    self.killed_score,
                                                                    obs.observation['player'][_ARMY_CAP_USED] /
                                                                    obs.observation['player'][_ARMY_CAP],
                                                                    obs.observation["score_cumulative"][0] / 35))

            self.ddqn.remember(self.previous_state, self.previous_action, self.reward, 'terminal', True)

            self.previous_action = None
            self.previous_state = None
            self.smart_action = None

            self.killed_score = 0
            self.move_number = 0

            return actions.FunctionCall(_NO_OP, [])

        if self.move_number == 0:
            self.move_number += 1

            if killed_units > self.previous_killed:
                self.killed_score += 0.25

            current_state = np.zeros(3)
            current_state[0] = obs.observation['player'][_ARMY_CAP_USED] / obs.observation['player'][_ARMY_CAP]
            current_state[1] = obs.observation["score_cumulative"][0] / 35
            current_state[2] = self.killed_score

            if self.previous_action is not None:
                self.ddqn.remember(self.previous_state, self.previous_action, self.reward, current_state, False)

            rl_action = self.ddqn.act(current_state)
            # print('Action:', rl_action)
            if len(re.findall("[0-9]", str(rl_action))) > 0:
                self.smart_action = smart_actions[int(rl_action)]
            else:
                self.smart_action = smart_actions[smart_actions.index(str(rl_action))]

            self.previous_state = current_state
            self.previous_action = self.smart_action
            self.previous_killed = killed_units

            if self.smart_action == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP, [])

            elif self.smart_action == ACTION_BUILD_MARINE or self.smart_action == ACTION_BUILD_GHOST or \
                    self.smart_action == ACTION_BUILD_MARAUDER or self.smart_action == ACTION_BUILD_REAPER:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif self.smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number = 0
            self.reward -= 1

            if self.smart_action == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP, [])

            elif self.smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif self.smart_action == ACTION_BUILD_GHOST:
                if _TRAIN_GHOST in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_GHOST, [_QUEUED])

            elif self.smart_action == ACTION_BUILD_MARAUDER:
                if _TRAIN_MARAUDER in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARAUDER, [_QUEUED])

            elif self.smart_action == ACTION_BUILD_REAPER:
                if _TRAIN_REAPER in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_REAPER, [_QUEUED])

            elif self.smart_action == ACTION_ATTACK:
                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [48, 36]])

        return actions.FunctionCall(_NO_OP, [])
