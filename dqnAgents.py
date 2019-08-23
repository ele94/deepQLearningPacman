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
from skimage import color
import tensorflow as tf

from datetime import datetime
from keras import losses
from PIL import Image
import gc, pickle, copy

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

        #gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | gc.DEBUG_OBJECTS) #debugging for memory leaks
        #print "IS GC ENABLED: ", gc.isenabled()
        self.action_size = 5
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.count = 0
        self.init = 0
        self.epsilon = 1.0
        self.alpha = 0.001
        self.discount = 0.99
        self.batch_size = 35
        self.frame_width = 85
        self.frame_height = 85
        self.state_size = 3
        self.image = None
        self.new_episode = True
        self.training = True
        self.best_score = -10000

        self.memory = deque(maxlen=100000)
        self.model = self._build_model()
        self.double_model = copy.deepcopy(self.model)

        #self.model.load_weights("models/best/model.h5") #uncomment to load old weights
        #self.double_model.load_weights("models/best/model.h5")

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
        action = None

        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return action  # Terminal State, return None

        #if self.image is None: # we only need to compute it the first time. afterwards, well get a copy from nextframe
        if self.new_episode:
            # self.new_episode = True
            self.image = self.process_frame(getFrame())
            self.new_episode = False

        # epsilon greedy: exploit - explore
        if self.epsilon > random.random():
            action = random.choice(legalActions)  # Explore
        else:
            act_values = self.model.predict(self.image/255.0) # Exploit

            for value in act_values[0]:
                action = PACMAN_ACTIONS[(np.argmax(act_values[0]))]
                if action not in legalActions:
                    act_values[0][np.argmax(act_values[0])] = -10000
                else:
                    break

            if action not in legalActions:
                print 'action was not legal'
                action = 'Stop'

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
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(nextState):
            done = True
        else:
            done = False

        if not done:
            self.nextImage = self.process_frame(getFrame())
        else:
            self.nextImage = None

        if not self.new_episode: # saving the first episode gives us trouble, so we skip it
            self.remember(self.image, action, reward, self.nextImage, done)

        # entrenamos la red neuronal solo mientras estamos en entrenamiento
        #if self.episodesSoFar < self.numTraining:
        #if self.episodesSoFar < self.numTraining:

        self.count += 1
        if len(self.memory) > 5*self.batch_size and self.training: # we only update the network while it's training
            self.replay(self.batch_size)

        # we save the model and deque every 1000 steps for safekeeping
        if self.count % 1000 == 0: #1000
            self.model.save_weights("models/model.h5")
            print "saved file: models/model.h5"
            self.double_model.save_weights("models/double_model.h5")

        # we save the frames every 10000 steps for debugging purposes
        if self.count % 10000 == 0 and not self.new_episode:
            img = Image.fromarray(self.image[0], 'RGB')
            img.save("frames/"+str(datetime.now())+"image.png")
            if not self.nextImage is None:
                img = Image.fromarray(self.nextImage[0], 'RGB')
                img.save("frames/" + str(datetime.now()) + "nextimage.png")

        self.image = copy.deepcopy(self.nextImage) #updating old image

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementAgent.final(self, state)
        #gc.collect() #por si escaso
        #del gc.garbage[:] # por si escaso
        print "memory length: ", len(self.memory)
        self.new_episode = True
        if self.training and state.getScore() > self.best_score:
            self.model.save_weights("models/best_model.h5")
            self.double_model.save_weights("models/best_dobule_model.h5")
            self.best_score = state.getScore()
            print "updated best models for best score: ", self.best_score
        if self.episodesSoFar % 1500 == 0:
            print "starting memory dump"
            with open('models/memory.dictionary', 'wb') as memory_deque_file:
                pickle.dump(self.memory, memory_deque_file)
            print "finished memory dump"
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            self.training = False
            pass

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu',
        #                        input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
                                 input_shape=(self.frame_height, self.frame_width, self.state_size)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        #model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        model.compile(loss='logcosh', optimizer='adam', metrics= ["accuracy"])
        return model

    def remember(self, state, action, reward, next_state, done):
        action_index = PACMAN_ACTIONS.index(action)
        #reward = self.normalize_reward(reward)
        if state is not None:
            self.memory.append((state, action_index, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, batch_size)
        for state, action_index, reward, next_state, done in minibatch:

            state=state/255.0
            if next_state is not None:
                next_state=next_state/255.0

            y_target = self.model.predict(state)
            y_target[0][action_index] = reward if done else reward + self.discount * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        #self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        self.double_model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        # we update the double network weights to the main network every 100 steps
        if self.count % 100 == 0:
            self.double_model.save_weights("models/double_model.h5")
            self.model.load_weights("models/double_model.h5")

        if self.epsilon > self.epsilon_min: # TODO: add timer before starting decay?
            self.epsilon *= self.epsilon_decay

    ##### HELPER METHODS #########
    def normalize_reward(self,reward):
        return np.sign(reward)

    def process_frame(self, frame):
        if frame is None:
            return None
        frame = np.array(frame)
        frame = resize(frame, (self.frame_width, self.frame_height))
        frame = np.reshape(frame, [1, self.frame_height, self.frame_width, self.state_size])
        frame = 255 * frame
        frame = np.uint8(frame)
        return frame

    # method for debugging
    def test(self):
        print self.model.predict(self.image/255.0)
        self.replay(1)
        print self.model.predict(self.image/255.0)
