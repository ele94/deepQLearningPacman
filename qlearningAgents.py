# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math, pickle

class QLearningAgent(ReinforcementAgent):
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
        self.QValueCounter = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValueCounter[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return 0

        best_action = self.computeActionFromQValues(state)
        return self.getQValue(state, best_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): return None

        best_action = None;
        best_value = float('-inf')
        for action in self.getLegalActions(state):
            if self.getQValue(state, action) > best_value:
                best_value = self.getQValue(state, action)
                best_action = action
        return best_action

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
        if not self.getLegalActions(state): return None  # Terminal State, return None

        if self.epsilon > random.random():
            action = random.choice(legalActions)  # Explore
        else:
            action = self.computeActionFromQValues(state)  # Exploit

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
        best_action = self.computeActionFromQValues(nextState)
        self.QValueCounter[(state, action)] = ((1 - self.alpha) * self.getQValue(state, action) +
                                               self.alpha * (reward + self.discount * self.getQValue(nextState,
                                                                                                     best_action)))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class MyPacmanQAgent(PacmanQAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

        # loading an existing q-table
        # self.loadTableFromFile('approxQweights.pkl')

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

    def saveTableToFile(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.QValueCounter, output, pickle.HIGHEST_PROTOCOL)

    def loadTableFromFile(self, filename):
        with open(filename, 'rb') as input:
            self.QValueCounter = pickle.load(input)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        print "Dictionary length: ", len(self.QValueCounter)
        print
        print
        print

        if self.episodesSoFar <= self.numTraining:
            self.saveTableToFile(self, 'qvalues.pkl')
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass



class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()


    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_dictionary = self.featExtractor.getFeatures(state, action)

        QValue = 0
        for feature in feature_dictionary:
            QValue += self.weights[feature] * feature_dictionary[feature]
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feature_dictionary = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)

        for feature in feature_dictionary:
            self.weights[feature] += self.alpha * difference * feature_dictionary[feature]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class MyApproximateQAgent(PacmanQAgent):

    def __init__(self, extractor='MyFeatureExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        #self.epsilon = 1.0
        #self.epsilon_min = 0.1
        #self.epsilon_decay = 0.99
        #
        self.weights = util.Counter()

        # loading existing weights
        # self.loadWeightsFromFile('weights/smallgridqweights.pkl')

    def getWeights(self):
        return self.weights

    def saveWeightsToFile(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.weights, output, pickle.HIGHEST_PROTOCOL)

    def loadWeightsFromFile(self, filename):
        with open(filename, 'rb') as input:
            self.weights = pickle.load(input)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_dictionary = self.featExtractor.getFeatures(state, action)

        QValue = 0
        for feature in feature_dictionary:
            QValue += self.weights[feature] * feature_dictionary[feature]
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feature_dictionary = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)

        for feature in feature_dictionary:
            self.weights[feature] += self.alpha * difference * feature_dictionary[feature]

        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        if self.episodesSoFar <= self.numTraining:
            self.saveWeightsToFile('approxQweights.pkl')

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass