import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
import numpy
import theano.tensor as T
import theano
from theano import function, shared

from deeplearning.mlp import MLP

##
n_hidden = 5
discount_factor = 0.99
learning_rate = 0.2
p_exploration = 0
p_exploration_decay = 0.5
##

# use double-precision for convenience
theano.config.floatX = 'float64'

def make_rlglue_action(action):
    a = Action()
    a.intArray = [int(action)]
    return a

class Experience(object):
    def __init__(self, prev_state, prev_action, state, reward):
        self.prev_state = prev_state
        self.prev_action = prev_action
        self.state = state
        self.reward = reward

class mlp_agent(Agent):
    """
    A Q-learning agent using a feedforward MLP to represent the Q function.
    The network is trained using Neural Fitted Q Iteration and Rprop.

    Greg Maslov <maslov@cs.unc.edu>
    """

    def __init__(self):
        self.numpy_rng = numpy.random.RandomState(42)

        self.state_size = None
        self.action_size = 1

        self.state_bounds = None
        self.num_actions = 3
        self.reward_bounds = None

        self.mlp = None

        self.prev_state = None
        self.prev_action = None

        self.p_exploration = p_exploration

    def agent_init(self, spec):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(spec)
        assert TaskSpec.valid, "TaskSpec could not be parsed: " + spec

        assert len(TaskSpec.getIntObservations())==0, "expecting continuous observations only"
        assert len(TaskSpec.getDoubleActions())==0, "expecting discrete actions only"
        self.state_size = len(TaskSpec.getDoubleObservations())
        self.action_size = len(TaskSpec.getIntActions())
        assert self.action_size == 1, 'expecting one action dimension only'
        
        specObs = TaskSpec.getDoubleObservations()
        self.state_bounds = []
        for i in xrange(0,self.state_size):
            self.state_bounds += [(specObs[i][0],specObs[i][1])]
        specAct = TaskSpec.getIntActions()
        self.action_bounds = []
        for i in xrange(0,self.action_size):
            self.action_bounds += [(specAct[i][0],specAct[i][1])]
        assert self.action_bounds[0][0]==0, 'action indices should start at 0'
        self.num_actions = self.action_bounds[0][1] - self.action_bounds[0][0] + 1
        self.reward_bounds = TaskSpec.getRewardRange()[0]

        print('compiling model...')
        self.x_state = T.matrix('x_state')
        self.x_action = T.ivector('x_action')
        onehotifier = shared(numpy.arange(self.num_actions,dtype='int32').reshape((self.num_actions,1)), broadcastable=(False,True))
        self.x_action_onehot = T.eq(onehotifier, self.x_action).dimshuffle((1,0))
        self.x = T.concatenate([self.x_state, self.x_action_onehot],axis=1)

        #def output_activation(x): return (1/(1-discount_factor))*(T.tanh(x)*(self.reward_bounds[1]-self.reward_bounds[0])+self.reward_bounds[0])
        #self.mlp = MLP(rng=self.numpy_rng, input=self.x, n_in=self.state_size+self.num_actions, n_hidden=n_hidden, n_out=1, activation=output_activation)
        # Linear output activation
        self.mlp = MLP(rng=self.numpy_rng, input=self.x, n_in=self.state_size+self.num_actions, n_hidden=n_hidden, n_out=1, activation=None)
        self.q = self.mlp.output[:,0]
        self.evaluate = function([self.x_state,self.x_action], self.q)

        max_state = T.vector('state')
        max_state_repeated = T.concatenate(self.num_actions*[max_state.dimshuffle('x',0)],axis=0)
        # Return the Q value and identity of the maximum-Q action from a given state.
        self.max_action = function([max_state],
                T.max_and_argmax(self.q),
                givens = {self.x_action: numpy.arange(self.num_actions,dtype='int32'),
                          self.x_state: max_state_repeated})

        self.update = self.compile_rprop_update()

        self.experiences = []

    def compile_rprop_update(self):
        """Returns a function implementing one iteration of Rprop training."""
        eta_n = 0.5
        eta_p = 1.2
        v_min = 1e-6
        v_max = 50

        self.rprop_values = [shared(learning_rate*numpy.ones(p.get_value(borrow=True).shape)) for p in self.mlp.params]
        self.rprop_signs = [shared(numpy.zeros(p.get_value(borrow=True).shape)) for p in self.mlp.params]
        target = T.vector('target')
        # L2-norm regularization term keeps the weight magnitudes under control
        cost = T.sum(T.sqr(self.q - target)) + 0.001*self.mlp.L2_sqr

        updates = []
        for p,v,s in zip(self.mlp.params,self.rprop_values,self.rprop_signs):
            g = T.grad(cost, p)
            s_new = T.sgn(g)
            sign_changed = T.neq(s, s_new)
            updates.append((p, T.switch(sign_changed, p, p - v*s_new)))
            updates.append((v, T.clip(T.switch(sign_changed, eta_n*v, eta_p*v),v_min,v_max)))
            updates.append((s, s_new))
        return function([self.x_state,self.x_action,target], T.sum(T.sqr(self.rprop_values[0])), updates=updates)

    def random_action(self):
        return random.choice(range(0,self.num_actions))

    # Observation -> Action
    def agent_start(self, observation):

        state = observation.doubleArray
        if random.random() < self.p_exploration:
            action = self.random_action()
        else:
            _, action = self.max_action(state)

        self.prev_state = copy.deepcopy(state)
        self.prev_action = copy.deepcopy(action)

        return make_rlglue_action(action)

    # R * Observation -> Action
    def agent_step(self, reward, observation):
        state = observation.doubleArray
        max_q, max_action = self.max_action(state)

        if random.random() < self.p_exploration:
            action = self.random_action()
        else:
            action = max_action

        self.experiences.append(Experience(self.prev_state,self.prev_action,state,reward))

        self.prev_state = copy.deepcopy(state)
        self.prev_action = copy.deepcopy(action)

        return make_rlglue_action(action)

    # R -> ()
    def agent_end(self, reward):
        self.experiences.append(Experience(self.prev_state,self.prev_action,None,reward))

        states = numpy.vstack([e.prev_state for e in self.experiences])
        actions = numpy.array([e.prev_action for e in self.experiences],dtype='int32')
        targets = numpy.zeros(len(self.experiences))
        costs = []
        for n in xrange(10):
            # Recompute target Q values with current estimate
            for i in xrange(len(self.experiences)-1):
                max_q = self.max_action(self.experiences[i].state)[0]
                targets[i] = self.experiences[i].reward + discount_factor*max_q
            targets[-1] = self.experiences[i].reward

            cost = self.update(states,actions,targets)
            costs.append(cost)
        print 'Costs:',costs
        self.experiences = []

        self.p_exploration *= p_exploration_decay
        if self.p_exploration < 1:
            self.p_exploration = 0
        print 'p_exploration',self.p_exploration

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass

if __name__=="__main__":
    AgentLoader.loadAgent(mlp_agent())

