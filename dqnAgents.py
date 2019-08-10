from collections import deque

import numpy
from keras import Sequential
from keras.layers import Dense, Convolution2D, Flatten, np
from keras.optimizers import Adam

from game import *
from graphicsDisplay import getFrame, saveFrame
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math,keras
from keras.optimizers import SGD
from skimage.transform import resize
import tensorflow as tf

from datetime import datetime
from keras import losses
from PIL import Image

PACMAN_ACTIONS = ['North', 'South', 'East', 'West', 'Stop']


class PacmanDQNAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.state_size = 294
        #self.state_size = 1320
        self.action_size = 5
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.count = 0
        self.init = 0
        self.epsilon = 1.0
        self.alpha = 0.001
        #self.alpha = 1e-6
        self.discount = 0.99
        self.batch_size = 35
        self.alpha_decay = 0.01
        self.frame_width = 85
        self.frame_height = 85
        self.image = None
        self.new_episode = True

        self.memory = deque(maxlen=100000)
        #self.model = self._build_model()
        self.model = self.atari_model()

        print 'INIT'

        #numpy.set_printoptions(threshold=sys.maxsize)



    # def getQValue(self, state, action):
    #     """
    #       Returns Q(state,action)
    #       Should return 0.0 if we have never seen a state
    #       or the Q node value otherwise
    #     """
    #     "*** YOUR CODE HERE ***"
    #     #return self.QValueCounter[(state, action)]
    #
    #
    # def computeValueFromQValues(self, state):
    #     """
    #       Returns max_action Q(state,action)
    #       where the max is over legal actions.  Note that if
    #       there are no legal actions, which is the case at the
    #       terminal state, you should return a value of 0.0.
    #     """
    #     "*** YOUR CODE HERE ***"
    #     if not self.getLegalActions(state): return 0
    #
    #     best_action = self.computeActionFromQValues(state)
    #     return self.getQValue(state, best_action)
    #
    # def computeActionFromQValues(self, state):
    #     """
    #       Compute the best action to take in a state.  Note that if there
    #       are no legal actions, which is the case at the terminal state,
    #       you should return None.
    #     """
    #     "*** YOUR CODE HERE ***"
    #     if not self.getLegalActions(state): return None
    #
    #     best_action = None
    #     best_value = float('-inf')
    #     for action in self.getLegalActions(state):
    #         if self.getQValue(state, action) > best_value:
    #             best_value = self.getQValue(state, action)
    #             best_action = action
    #     return best_action


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        # if 'Stop' in legalActions:
        #     legalActions.remove('Stop')

        action = None
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return action  # Terminal State, return None

        if self.image is None: # we only need to compute it the first time. afterwards, well get the nextframe
            self.new_episode = True
            self.image = getFrame()
            self.image = np.array(self.image)
            self.image = resize(self.image, (self.frame_height, self.frame_width), order=0, anti_aliasing=True)
            #self.image = 255 * self.image
            #self.image = np.uint8(self.image)
            #self.image = np.reshape(self.image, [1,self.frame_height, self.frame_width,3])
            self.image = 255 * self.image
            self.image = np.uint8(self.image)
        else:
            self.new_episode = False
        #img = Image.fromarray(self.image[0], 'RGB')
        #img.save("frames/my"+str(datetime.now())+".png")

        #print 'Epsilon value: ', self.epsilon
        if self.epsilon > random.random():
            action = random.choice(legalActions)  # Explore
        else:
            #action = self.computeActionFromQValues(state)  # Exploit
            #state_matrix = self.getStateMatrices(state)
            #state_matrix = np.reshape(np.array(state_matrix), [1, self.state_size])
            #state_matrix = np.reshape(state_matrix, (1, self.frame_width, self.frame_height))

            #act_values = self.model.predict(self.image/255.0)
            self.testModel(self.image)
            print "-----------------------------------------"

            act_values = self.model.predict([np.reshape(self.image, [1,self.frame_height, self.frame_width,3])/255.0, np.ones((1,5))])
            print act_values
            print np.argmax(act_values[0])
            print PACMAN_ACTIONS[(np.argmax(act_values[0]))]
            print PACMAN_ACTIONS[np.argmax(act_values[0])]
            print PACMAN_ACTIONS[np.argmax(act_values)]
            for value in act_values[0]:
                action = PACMAN_ACTIONS[(np.argmax(act_values[0]))]
                if action not in legalActions:
                    act_values[0][np.argmax(act_values[0])] = -10000
                else:
                    break

            # action = PACMAN_ACTIONS[(np.argmax(act_values[0]))] # returns action
            # print 'chosen action: ', action
            #
            # if action not in legalActions:
            #     print 'action was not legal'
            #
            #     for value in act_values:
            #         act_values[0, np.argmax(act_values[0])] = -1000
            #         action = PACMAN_ACTIONS[(np.argmax(act_values[0]))]
            #         if action in legalActions:
            #             break
            #     print 'chose action: ', action
                #action = 'Stop'
                #action = random.choice(legalActions)

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        if not self.getLegalActions(nextState):
            done = True
        else:
            done = False
        "*** YOUR CODE HERE ***"

        if not done:
            self.nextImage = getFrame()
            self.nextImage = numpy.array(self.nextImage)
            self.nextImage = resize(self.nextImage, (self.frame_width, self.frame_height))
            #self.nextImage = np.reshape(self.nextImage, [1, self.frame_height, self.frame_width, 3])
            self.nextImage = 255 * self.nextImage
            self.nextImage = np.uint8(self.nextImage)
        else:
            self.nextImage = None
        #new code


        # state_matrix = self.getStateMatrices(state)
        # nextState_matrix = self.getStateMatrices(nextState)
        # state_matrix = np.reshape(state_matrix, [1, self.state_size])
        # nextState_matrix = np.reshape(nextState_matrix, [1, self.state_size])

        self.count += 1
        if not self.new_episode: #no guardamos la primera imagen porque da problemas
            self.remember(self.image, action, reward, self.nextImage, done)

        # entrenamos la red neuronal solo mientras estamos en entrenamiento
        #if self.episodesSoFar < self.numTraining:
        #if self.episodesSoFar < self.numTraining:


        if len(self.memory) > 1*self.batch_size:
            self.replay(self.batch_size)

        if self.count % 1000 == 0:
            self.model.save_weights("models/model" + str(self.count) + ".h5")
            print "saved file: models/model" + str(self.count) + ".h5"

        if self.count % 10000 == 0 and not self.new_episode:
            img = Image.fromarray(self.image[0], 'RGB')
            img.save("frames/"+str(datetime.now())+"image.png")
            if not self.nextImage is None:
                img = Image.fromarray(self.nextImage[0], 'RGB')
                img.save("frames/" + str(datetime.now()) + "nextimage.png")

        self.image = self.nextImage #updating old image


    # def getPolicy(self, state):
    #     return self.computeActionFromQValues(state)
    #
    # def getValue(self, state):
    #     return self.computeValueFromQValues(state)


    # def final(self, state):
    #     "Called at the end of each game."
    #     # call the super-class final method
    #     ReinforcementAgent.final(self, state)
    #
    #     state_matrix = self.getStateMatrices(state)
    #     nextState_matrix = self.getStateMatrices(self.lastState)
    #     state_matrix = np.reshape(state_matrix, [1, self.state_size])
    #     nextState_matrix = np.reshape(nextState_matrix, [1, self.state_size])
    #
    #     self.remember(state_matrix, action, reward, nextState_matrix, done)
    #
    #     # did we finish training?
    #     if self.episodesSoFar == self.numTraining:
    #         # you might want to print your weights here for debugging
    #         "*** YOUR CODE HERE ***"
    #         pass

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(48, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        model = Sequential()
        #STATE_LENGTH = self.state_size
        FRAME_WIDTH = self.frame_width
        FRAME_HEIGHT = self.frame_height
        STATE_LENGTH = 3
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
        #                        input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
                                 input_shape=(FRAME_HEIGHT, FRAME_WIDTH, STATE_LENGTH)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        #model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        model.compile(loss='mse', optimizer='adam', metrics= ["accuracy"])
        #model.compile(loss=losses.mean_squared_logarithmic_error, optimizer='adam', metrics= ["accuracy"])


        return model

    def remember(self, state, action, reward, next_state, done):
        action_index = PACMAN_ACTIONS.index(action)
        action_encoded = self.one_hot_encode(action_index)
        reward = transform_reward(reward)
        # if next_state is None:
        #     next_state = state
        print action, reward, done
        self.memory.append((state, action_encoded, reward, next_state, done))


    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in minibatch:

            # print 'state: ', state
            # print 'action_index: ', action_index
            # print 'reward: ', reward
            # print 'next_state: ', next_state
            # print 'done: ', done
            # img = Image.fromarray(state[0], 'RGB')
            # img.save("frames/my"+str(datetime.now())+".png")
            # img = Image.fromarray(next_state[0], 'RGB')
            # img.save("frames/my"+str(datetime.now())+".png")
            if next_state is None:
                next_state = state
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            # state=state/255.0
            # if next_state is not None:
            #     next_state=next_state/255.0
            # y_target = self.model.predict(state)
            # y_target[0][action_index] = reward if done else reward + self.discount * np.max(self.model.predict(next_state)[0])
            # x_batch.append(state[0])
            # y_batch.append(y_target[0])
            #
            # next_Q_values = self.model.predict([state, np.ones(actions.shape)])
            # next_Q_values[done] = 0

        #self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        #W_Input_Hidden = self.model.layers[0].get_weights()[0]
        #print W_Input_Hidden
        minibatch = np.array(minibatch)
        # print type(minibatch[:,0])
        # print minibatch[:,4]
        #self.fit_batch(minibatch[:,0], minibatch[:,1], minibatch[:,2], minibatch[:,3], minibatch[:,4])
        self.fit_batch(np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones))
        #self.fit_batch(minibatch)

        if self.epsilon > self.epsilon_min:
            #None
            self.epsilon *= self.epsilon_decay


    def fit_batch(self, start_states, actions, rewards, next_states, is_terminal):
        """Do one deep Q learning iteration.
    
        Params:
        - model: The DQN
        - gamma: Discount factor (should be 0.99)
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal
    
        """
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        #print actions.shape
        print actions.shape
        next_Q_values = self.model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + self.discount * np.max(next_Q_values, axis=1)
        #print Q_values
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.model.fit(
            [start_states, actions], actions * Q_values[:, None],
            nb_epoch=1, batch_size=len(start_states), verbose=0
        )


    def one_hot_encode(self,action_index):
        actions = np.zeros(5)
        actions[action_index] = 1
        return actions

    def atari_model(self):
        # We assume a theano backend here, so the "channels" are first.
        n_actions=self.action_size
        ATARI_SHAPE = (85, 85, 3)

        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)

        conv_1 = keras.layers.convolutional.Convolution2D(
            16, 8, 8, subsample=(4, 4), activation='relu'
        )(normalized)
        conv_2 = keras.layers.convolutional.Convolution2D(
            32, 4, 4, subsample=(2, 2), activation='relu'
        )(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        #filtered_output = keras.layers.merge([output, actions_input], mode='mul')
        filtered_output = keras.layers.Multiply()([output, actions_input])

        self.model = keras.models.Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
        return self.model

    def testModel(self, image):
        act_values = self.model.predict(
            [np.reshape(image, [1, self.frame_height, self.frame_width, 3]) / 255.0, np.ones((1, 5))])
        print act_values
        self.replay(1)
        act_values = self.model.predict(
            [np.reshape(image, [1, self.frame_height, self.frame_width, 3]) / 255.0, np.ones((1, 5))])
        print act_values

def transform_reward(reward):
    return np.sign(reward)

