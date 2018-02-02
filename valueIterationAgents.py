# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        mdpStates = self.mdp.getStates()
    
        valuesDic = {} #temporary dictionary
        for iteration in range(self.iterations):
          # copy self.values into a temporary dictionary, valuesDic
          valuesDic = dict(self.values)
          for state in mdpStates:
            qValuesList = [] 
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              #qValuesList: a list of qValues computed with all the possible actions from the current state
              actionQValue = self.computeQValueFromValues(state,action)
              qValuesList += [actionQValue]

            #There was no possible actions  
            if len(qValuesList) == 0:
              valuesDic[state] = 0
            else:
              #Update the key of the current state in valuesDic to have the maximum Qvalue
              valuesDic[state] = max(qValuesList)

          self.values = valuesDic

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Q(s,a) = sum{T(s,a,s')[R(s,a,s') + gamma*V(s')]}

        gamma = self.discount  #discount factor
        #statesAndProbs: a list of (transition state, probability of the transition state)
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        sumQvalues = 0

        for (transState, prob) in statesAndProbs:
          reward = self.mdp.getReward(state,action,transState)
          transStateValue = self.getValue(transState)
          #Q(s,a) is a cumulative value
          sumQvalues += prob*(reward+(gamma*transStateValue))

        return sumQvalues

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state) #possible actions

        # There's no possible next action
        if len(actions) <= 0:
          return None

        bestAction = None
        bestValue = float("-inf")
        for action in actions:
          qValue = self.getQValue(state, action)
       
          if bestValue < qValue:
            #update bestValue to be the current qValue so that bestValue is the maximum value at the end
            bestValue = qValue
            #update bestAction to be the current action
            bestAction = action
            
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        

    def runValueIteration(self):
        mdpStates = self.mdp.getStates()

        for iteration in range(self.iterations):
          if iteration < len(mdpStates):
            state = mdpStates[iteration]
          else:
            #print "iteration mod len(mdpStates): ", iteration % len(mdpStates) 
            state = mdpStates[iteration % len(mdpStates)] #To prevent index out of range
        
          if not self.mdp.isTerminal(state):
            qValuesList = [] 
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
              #qValuesList: a list of qValues computed with all the possible actions from the current state
              actionQValue = self.computeQValueFromValues(state,action)
              qValuesList += [actionQValue]
            #Update the current state in self.values to have the maximum Qvalue as its value
            self.values[state] = max(qValuesList)
   
          else:
            #TERMINAL STATE
            self.values[state] = self.values[state]
          

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pQueue = util.PriorityQueue() #Initialize a priority queue
        mdpStates = self.mdp.getStates()
        
        predecessorDic = {} #key:state, value:a set of predecessors 
        for state in mdpStates:
          predecessorDic[state]=set() #Using a set, instead of a list, to avoid duplicates

        for state in mdpStates:
          actions = self.mdp.getPossibleActions(state)
          for action in actions:
            #statesAndProbs: a list of (transition state, probability of the transition state)
            statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            for (transState, prob) in statesAndProbs:
              if prob != 0:
                # If the state can take action(s) to get to the next state with a 
                #  nonzero probability, it is a predecessor of the next state
                predecessorDic[transState].add(state)
                
          if not self.mdp.isTerminal(state):
            # Find the highest Q-value across all the possible actions from the state
            qValuesList = []
            qValuesList += [self.getQValue(state,action) for action in actions]
            maxQValue = max(qValuesList)

            currentValue = self.values[state]
            diff = abs(maxQValue-currentValue)
            # We take the negative of diff because we want to prioritize updating 
            #  states that have a higher error.
            pQueue.push(state, -diff)

        for iteration in range(self.iterations):
          if pQueue.isEmpty(): 
            break #terminate if the priority queue is empty
          else:
            popState = pQueue.pop()
            actions = self.mdp.getPossibleActions(popState)
            #Update self.values for the state just popped with the maximum Q-Value across all the possible actions
            self.values[popState] = max(self.getQValue(popState,action) for action in actions)

            predecessorSet = predecessorDic[popState] #A set of predecessors to popState
            for predState in predecessorSet:
              pred_qValuesList = []
              actions = self.mdp.getPossibleActions(predState)

              # Find the highest Q-value across all the possible actions from popState
              pred_qValuesList += [self.getQValue(predState,action) for action in actions]
              pred_max_QValue = max(pred_qValuesList)

              currentValue = self.values[predState]
              diff = abs(pred_max_QValue - currentValue)

              if diff > self.theta:
                #If predState in pQueue does not already have a priority lower or equal,
                # update pQueue with a new priority 
                pQueue.update(predState, -diff)
