import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

from random import Random

class random_agent(Agent):
    rng = Random()

    state_size = 0
    action_size = 0

    state_bounds = []
    action_bounds = []

    #lastAction=Action()
    #lastObservation=Observation()
    
    def agent_init(self, spec):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(spec)
        assert TaskSpec.valid, "TaskSpec could not be parsed: " + spec

        assert len(TaskSpec.getIntObservations())==0, "expecting continuous observations only"
        assert len(TaskSpec.getDoubleActions())==0, "expecting discrete actions only"
        self.state_size = len(TaskSpec.getDoubleObservations())
        self.action_size = len(TaskSpec.getIntActions())
        
        specObs = TaskSpec.getDoubleObservations()
        self.state_bounds = []
        for i in xrange(0,self.state_size):
            self.state_bounds += [(specObs[i][0],specObs[i][1])]
        specAct = TaskSpec.getIntActions()
        self.action_bounds = []
        for i in xrange(0,self.action_size):
            self.action_bounds += [(specAct[i][0],specAct[i][1])]

    def agent_start(self, observation):
        #Generate random action, 0 or 1
        return_action = Action()
        return_action.intArray = []
        for i in xrange(0,self.action_size):
            return_action.intArray += [self.rng.randint(self.action_bounds[i][0],self.action_bounds[i][1])]
        return return_action
        
        #lastAction=copy.deepcopy(returnAction)
        #lastObservation=copy.deepcopy(observation)
    
    def agent_step(self, reward, observation):
        return self.agent_start(observation)
    
    def agent_end(self, reward):
        pass
    
    def agent_cleanup(self):
        pass
    
    def agent_message(self, inMessage):
        return "I don't know how to respond to your message";

if __name__=="__main__":
    AgentLoader.loadAgent(random_agent())

