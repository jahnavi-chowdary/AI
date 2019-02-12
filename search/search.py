# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    #implement using stack
    "*** YOUR CODE HERE ***"
    #start - get the start state
    start = problem.getStartState()
    #stack to store the fringe list
    stk = util.Stack()
    stk.push((start,[]))
    #visited array to store the expanded nodes
    visited = []
    while not stk.isEmpty():
        #curr - stores the current state of the problem 
        #direc - the list of directions containg the path to the goal state
        curr, direc = stk.pop()       
        if problem.isGoalState(curr):
            return direc
        curr_list = problem.getSuccessors(curr)
        #curr_list.reverse()
        for succ, act, stp_cost in curr_list:
            if not succ in visited:
                stk.push((succ, direc + [act]))
                #appending the visited(expanded) list
                visited.append(curr)
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    "*** YOUR CODE HERE ***"
    #start - the starting posotion of the pacman
    start = problem.getStartState()
    #We use Queue in BFS as a fringe list.
    q = util.Queue()
    q.push((start,[]))
    visited = []
    while not q.isEmpty():
        curr, direc = q.pop()
        if not curr in visited:
            visited.append(curr)
            if problem.isGoalState(curr):
                return direc
            for succ, act, stp_cost in problem.getSuccessors(curr):
                q.push((succ, direc + [act]))
    return []
    """Search the shallowest nodes in the search tree first."""
    util.raiseNotDefined()
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #start - the starting posotion of the pacman
    start = problem.getStartState()
    #Using priority queue here as we have to consider the costs of the path as well
    #We use least cost path here
    q = util.PriorityQueue()
    q.push((start,[]), 0)
    visited = []
    while not q.isEmpty():
        #curr, direc, stp_cost = q.pop()
        curr, direc = q.pop()
        if not curr in visited:
            visited.append(curr)
            if problem.isGoalState(curr):
                return direc
            for succ, act, stp in problem.getSuccessors(curr):
                #q.push((succ, act, stp+stp_cost), stp+stp_cost)
                q.push((succ, direc + [act]), problem.getCostOfActions(direc + [act]))
                #q.update((succ, act, stp+stp_cost), stp+stp_cost)
                q.update((succ, direc + [act]), problem.getCostOfActions(direc + [act]))
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    q = util.PriorityQueue()
    q.push((start,[]), 0)
    visited = []
    while not q.isEmpty():
        #curr, direc, stp_cost = q.pop()
        curr, direc = q.pop()
        if not curr in visited:
            visited.append(curr)
            if problem.isGoalState(curr):
                return direc
            for succ, act, stp in problem.getSuccessors(curr):
                #q.push((succ, act, stp+stp_cost), stp+stp_cost)
                q.push((succ, direc + [act]), \
                    problem.getCostOfActions(direc + [act])+heuristic(succ, problem))
                #q.update((succ, act, stp+stp_cost), stp+stp_cost)
                q.update((succ, direc + [act]), \
                    problem.getCostOfActions(direc + [act])+heuristic(succ, problem))
    return []
    """Search the node that has the lowest combined cost and heuristic first."""
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
