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

import random,util,math

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
        ReinforcementAgent.__init__(self, **args)
        #A dictionay with key as (state,action) and value as the Q-Value 
        # computed for the key state and the key action
        self.qValues = util.Counter() # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state,action) in self.qValues:
          return self.qValues[(state,action)]
        else:
        # We've never seen this state, return 0.0
          return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        # If there are no legal actions, state is Terminal. Return 0.0
        if len(actions) == 0: 
          return 0.0
        else:
          # maximum Q-Value from all the Q-Values across all the legal actions 
          #  that can be taken from the state
          maxQValue = max([self.getQValue(state, action) for action in actions])
          return maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        # If there are no legal actions, state is Terminal. Return None.
        if len(actions) == 0:
          return None
        else:
          # actions_qVals is a dict with key as action and value as its Q-Value at the given state
          actions_qVals = util.Counter() # A Counter is a dict with default 0
          for action in actions:
            if (state, action) in self.qValues:
              actions_qVals[action] = self.qValues[(state,action)]

          best_value = actions_qVals[actions_qVals.argMax()] #max Q-value
          # best_actions is a list of all the actions with maximum Q-Values
          best_actions = []
          for action in actions:
            if actions_qVals[action] == best_value:
              best_actions += [action]

          # best_action can be chosen randomly from best_actions, because
          #  all the actions in this list are tied in terms of their Q-values.
          best_action = random.choice(best_actions)

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
        # Epsilon_Greedy
        legalActions = self.getLegalActions(state)
        action = None

        #If there are no legal actions, state is Terminal. Return None.
        if len(legalActions) == 0:
          action = None
        # Take a random action
        elif util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          # Take the best policy action
          action = self.getPolicy(state)
    
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Q(s,a) = (1-alpha)* old_Q(s,a) + alpha*[R(s,a,s')+gamma*max{Q(s',a')}]

        gamma = self.discount #discount rate
        alpha = self.alpha #learning rate

        currentQValue = self.getQValue(state,action) # Old estimate

        nextActions = self.getLegalActions(nextState)
        #nextQValues is a list of the Q-Values across all the legal 
        # actions available at nextSate
        nextQValues = [] 
        for nextAction in nextActions:
          nextQValues += [self.getQValue(nextState, nextAction)]

        maxNextQVal = 0
        # Unless there are no legal actions that can be taken at the next state,
        #  find the maximum Q-Values across all the legal actions of nextState.
        if len(nextActions) != 0:
          maxNextQVal = max(nextQValues)
       
        # sample = R(s,a,s') + gamma*max{Q(s',a')}
        sample = reward + gamma*maxNextQVal
        newQValue = (1-alpha)*currentQValue + (alpha*sample)

        # Update the state's Q-Value based on learning about its next state
        self.qValues[(state,action)] = newQValue

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
        # Q(s,a) = sum(weight_i * f_i(s,a))
        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()
        #print weights
        sumVal = 0
        for feature in features.keys():
          featureVal = features[feature]
          sumVal += featureVal * weights[feature]
        return sumVal        

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        gamma = self.discount #discount rate
        alpha = self.alpha #learning rate
        features = self.featExtractor.getFeatures(state, action)

        currentQValue = self.getQValue(state,action) # Old estimate

        nextActions = self.getLegalActions(nextState)
        #nextQValues is a list of the Q-Values across all the legal 
        # actions available at nextSate
        nextQValues = [] 
        for nextAction in nextActions:
          nextQValues += [self.getQValue(nextState, nextAction)]

        maxNextQVal = 0
        # Unless there are no legal actions that can be taken at the next state,
        #  find the maximum Q-Values across all the legal actions of nextState.
        if len(nextActions) != 0:
          maxNextQVal = max(nextQValues)

        diff = reward + (gamma * maxNextQVal) - currentQValue

        for feature in features.keys():
          featureVal = features[feature]
          currentWeight = self.weights[feature]
          # Update the feature's weight in self.weights based on the learning 
          #  of weights for features of the next state
          self.weights[feature] = currentWeight + (alpha * diff * featureVal )

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
