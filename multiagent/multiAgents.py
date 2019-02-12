# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #print "leagal moves  ", legalMoves
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print " scores are ",scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #print "best indices ", bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print "chosen index  ", chosenIndex
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        #print "current game state ", currentGameState
        #print "action ", action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print "successor game state  ", successorGameState
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print "new scared times   ", newScaredTimes
        "*** YOUR CODE HERE ***"
        #calculate the food near to the pacman
        nearfood = currentGameState.getFood()
        currentPacmanPosition = list(newPos)
        maxdist = -1000000000
        #initialize dist variable to be 0
        dist = 0
        foodList = nearfood.asList()

        if action == 'Stop':
            return -1000000000
        #check the position of the ghosts
        for state in newGhostStates:
            if state.getPosition() == tuple(currentPacmanPosition) and (state.scaredTimer == 0):
                return -1000000000
        #check all the food distance by using manhattan distance and check for the max distance
        for foodpos in foodList:
            dist = manhattanDistance(foodpos, currentPacmanPosition)
            dist = -dist
            if (dist > maxdist):
                maxdist = dist
        return maxdist

        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isStateTerminal(self, state, depth, agent):
        return depth == self.depth or \
               state.isWin() or \
               state.isLose() or \
               state.getLegalActions(agent) == 0

    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, depth, agent):
            #if all the agents (ghosts and pacman have been visited)
            if agent == state.getNumAgents(): 
                return minimax(state, depth + 1, 0)
            #check if the terminal state has been reached
            if self.isStateTerminal(state, depth, agent):
                return self.evaluationFunction(state)
            list_succ = []
            for action in state.getLegalActions(agent):
              #generate the child nodes for all
              child = state.generateSuccessor(agent, action)
              val = minimax(child, depth, agent+1)
              list_succ.append(val)
            if self.isPacman(state, agent):
                return max(list_succ)
            else:
              return min(list_succ)
        pacman_state = 0
        var_legal = gameState.getLegalActions(pacman_state)
        return max(var_legal, key = lambda ans: minimax(gameState.generateSuccessor(0, ans), 0, 1)
        )
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def get_value(state, depth, agent, alpha, beta):
          # Check if it is a MAX Node
          if (agent == state.getNumAgents()):
            agent = 0 # Pacman Agent denoted with 0
            depth += 1 #Increment Depth
          
          # Check Base Case i.e check the terminal state
          if self.isStateTerminal(state, depth, agent):
            return self.evaluationFunction(state), None

          if (agent == 0):
            return get_maxvalue(state, depth, agent, alpha, beta) 
          else:
            return get_minvalue(state, depth, agent, alpha, beta)

        def get_maxvalue(state, depth, agent, alpha, beta):
          legalActions = state.getLegalActions(agent)
      
          # Initialize V and Action Output
          v = -1000000000
          actions_out = None

          for actions in legalActions:
            child = state.generateSuccessor(agent, actions)
            child_value,_ = get_value(child, depth, agent + 1, alpha, beta)
            
            # Update v and Actions Output
            if (v < child_value):
              actions_out = actions
            v = max(v, child_value)

            if (v > beta):
              return v, actions_out
            alpha = max(alpha, v)
          
          return v, actions_out

        def get_minvalue(state, depth, agent, alpha, beta):
          legalActions = state.getLegalActions(agent)
          
          # Initialize V and Action Output
          v = 1000000000
          actions_out = None

          for actions in legalActions:
            child = state.generateSuccessor(agent, actions)
            child_value,_ = get_value(child, depth, agent + 1, alpha, beta)
            
            # Update v and Actions Output
            if (v > child_value):
              actions_out = actions
            v = min(v, child_value)

            if (v < alpha):
              return v, actions_out
            beta = min(beta, v)
          
          return v, actions_out
        
        alpha = -1000000000
        beta = 1000000000
        depth = 0
        agent = 0
        _ , actions_out = get_value(gameState, depth, agent, alpha, beta)
        return actions_out
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agent):
            if agent == state.getNumAgents():
                return expectimax(state, depth + 1, 0)
            #check for terminal states
            if self.isStateTerminal(state, depth, agent):
                return self.evaluationFunction(state)
            #list_succ is a list of all the children of the current agent
            list_succ = []
            for action in state.getLegalActions(agent):
              child = state.generateSuccessor(agent, action)
              val = expectimax(child, depth, agent+1)
              list_succ.append(val)
            if self.isPacman(state, agent):
                return max(list_succ)
            else:
              return sum(list_succ)/len(list_succ)

        # return the best of pacman's possible moves
        pacman_state = 0
        var_legal = gameState.getLegalActions(pacman_state)
        return max(var_legal, key = lambda x: expectimax(gameState.generateSuccessor(0, x), 0, 1)
        )
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

