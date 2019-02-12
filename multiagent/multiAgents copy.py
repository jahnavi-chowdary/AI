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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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

        # MY CODE
        def get_value(state, depth, agent, alpha, beta):
          # Check if it is a MAX Node
          if (agent == state.getNumAgents()):
            agent = 0
            depth += 1
          
          # Check Base Case
          if self.isStateTerminal(state, depth, agent):
            return self.evaluationFunction(state), None

          if (agent == 0):
            return get_maxvalue(state, depth, agent, alpha, beta) 
          else:
            return get_minvalue(state, depth, agent, alpha, beta)

        def get_maxvalue(state, depth, agent, alpha, beta):
          legalActions = state.getLegalActions(agent)

          # Check Base Case
          #if (depth == self.depth or state.isWin() or state.isLose() or len(legalActions) == 0):
          #  return (self.evaluationFunction(state), None)
          
          # Initialize V and Action Output
          v = -float("inf")
          actions_out = None

          for actions in legalActions:
            successor_state = state.generateSuccessor(agent, actions)
            successor_value,_ = get_value(successor_state, depth, agent + 1, alpha, beta)
            
            if (v < successor_value):
              actions_out = actions
    
            v = max(v, successor_value)

            if (v > beta):
              return v, actions_out
          
            alpha = max(alpha, v)
          
          return v, actions_out

        def get_minvalue(state, depth, agent, alpha, beta):
          legalActions = state.getLegalActions(agent)

          # Check Base Case
          #if (len(legalActions) == 0):
          #  return (self.evaluationFunction(state), None)
          
          # Initialize V and Action Output
          v = float("inf")
          actions_out = None

          for actions in legalActions:
            successor_state = state.generateSuccessor(agent, actions)
            successor_value,_ = get_value(successor_state, depth, agent + 1, alpha, beta)
            
            if (v > successor_value):
              actions_out = actions
        
            v = min(v, successor_value)

            if (v < alpha):
              return v, actions_out
          
            beta = min(beta, v)
          
          return v, actions_out
        
        alpha = -(float("inf"))
        beta = float("inf")
        depth = 0
        agent = 0
        _, actions_out = get_value(gameState, depth, agent, alpha, beta)
        return actions_out

        """
        def dispatch(state, depth, agent, A=float("-inf"), B=float("inf")):
            if agent == state.getNumAgents():  # next depth
                depth += 1
                agent = 0

            if self.isTerminal(state, depth, agent):  # dead end
                return self.evaluationFunction(state), None

            if self.isPacman(state, agent):
                return getValue(state, depth, agent, A, B, float('-inf'), max)
            else:
                return getValue(state, depth, agent, A, B, float('inf'), min)

        def getValue(state, depth, agent, A, B, ms, mf):
            bestScore = ms
            bestAction = None

            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                score,_ = dispatch(successor, depth, agent + 1, A, B)
                bestScore, bestAction = mf((bestScore, bestAction), (score, action))

                if self.isPacman(state, agent):
                    if bestScore > B:
                        return bestScore, bestAction
                    A = mf(A, bestScore)
                else:
                    if bestScore < A:
                        return bestScore, bestAction
                    B = mf(B, bestScore)

            return bestScore, bestAction

        _,action = dispatch(gameState, 0, 0)
        return action
        """

        """
        " Max value "
        def max_value(gameState, depth, alpha, beta):
            " Cases checking "
            actionList = gameState.getLegalActions(0) # Get actions of pacman
            if len(actionList) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            
            " Initializing the value of v and action to be returned "
            v = -(float("inf"))
            goAction = None

            for thisAction in actionList:
                successorValue = min_value(gameState.generateSuccessor(0, thisAction), 1, depth, alpha, beta)[0]
                " v = max(v, successorValue) "
                if (v < successorValue):
                    v, goAction = successorValue, thisAction

                if (v > beta):
                    return (v, goAction)

                alpha = max(alpha, v)

            return (v, goAction)

        " Min value "
        def min_value(gameState, agentID, depth, alpha, beta):
            " Cases checking "
            actionList = gameState.getLegalActions(agentID) # Get the actions of the ghost
            if len(actionList) == 0:
              return (self.evaluationFunction(gameState), None)

            " Initializing the value of v and action to be returned "
            v = float("inf")
            goAction = None

            for thisAction in actionList:
                if (agentID == gameState.getNumAgents() - 1):
                    successorValue = max_value(gameState.generateSuccessor(agentID, thisAction), depth + 1, alpha, beta)[0]
                else:
                    successorValue = min_value(gameState.generateSuccessor(agentID, thisAction), agentID + 1, depth, alpha, beta)[0]
                " v = min(v, successorValue) "
                if (successorValue < v):
                    v, goAction = successorValue, thisAction

                if (v < alpha):
                    return (v, goAction)

                beta = min(beta, v)

            return (v, goAction)

        alpha = -(float("inf"))
        beta = float("inf")
        return max_value(gameState, 0, alpha, beta)[1]
        util.raiseNotDefined()
        """

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

