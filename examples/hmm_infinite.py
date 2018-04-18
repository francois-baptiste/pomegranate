
# coding: utf-8

# # Infinite Hidden Markov Model

# authors:<br>
# Jacob Schreiber [<a href="mailto:jmschreiber91@gmail.com">jmschreiber91@gmail.com</a>]<br>
# Nicholas Farn [<a href="mailto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# This example shows how to use yahmm to sample from an infinite HMM. The premise is that you have an HMM which does not have transitions to the end state, and so can continue on forever. This is done by not adding transitions to the end state. If you bake a model with no transitions to the end state, you get an infinite model, with no extra work! This change is passed on to all the algorithms.

# In[1]:
from pomegranate.base import State
from pomegranate.distributions import NormalDistribution
from pomegranate.hmm import HiddenMarkovModel

from pomegranate import *
import itertools as it
import numpy as np


# First we define the possible states in the model. In this case we make them all have normal distributions.

# In[2]:


s1 = State( NormalDistribution( 5, 2 ), name="S1" )
s2 = State( NormalDistribution( 15, 2 ), name="S2" )
s3 = State( NormalDistribution( 25, 2 ), name="S3" )


# We then create the HMM object, naming it, logically, "infinite".

# In[3]:


model = HiddenMarkovModel( "infinite" )


# We then add the possible transition, making sure not to add an end state. Thus with no end state, the model is infinite!

# In[4]:


model.add_transition( model.start, s1, 0.7 )
model.add_transition( model.start, s2, 0.2 )
model.add_transition( model.start, s3, 0.1 )
model.add_transition( s1, s1, 0.6 )
model.add_transition( s1, s2, 0.1 )
model.add_transition( s1, s3, 0.3 )
model.add_transition( s2, s1, 0.4 )
model.add_transition( s2, s2, 0.4 )
model.add_transition( s2, s3, 0.2 )
model.add_transition( s3, s1, 0.05 )
model.add_transition( s3, s2, 0.15 )
model.add_transition( s3, s3, 0.8 )


# Finally we "bake" the model, finalizing the model.

# In[5]:


model.bake()


# Now we can check whether or not our model is infinite.

# In[6]:


#print(model.is_infinite())


# Now lets the possible states in the model.

# In[7]:


print("States")
print("\n".join( state.name for state in model.states ))


# Now lets test out our model by feeding it a sequence of values. We feed our sequence of values first through a forward algorithm in our HMM.

# In[8]:


sequence = [ 4.8, 5.6, 24.1, 25.8, 14.3, 26.5, 15.9, 5.5, 5.1 ]

print("Forward")
print(model.forward( sequence ))


# That looks good as well. Now lets feed our sequence into the model through a backwards algorithm.

# In[9]:


print("Backward")
print(model.backward( sequence ))


# Continuing on we now feed the sequence in through a forward-backward algorithm.

# In[10]:


print("Forward-Backward")
trans, emissions = model.forward_backward( sequence )
print(trans)
print(emissions)


# Finally we feed the sequence through a Viterbi algorithm to find the most probable sequence of states.

# In[11]:


print("Viterbi")
prob, states = model.viterbi( sequence )
print("Prob: {}".format( prob ))
print("\n".join( state[1].name for state in states ))
print()
print("MAP")
prob, states = model.maximum_a_posteriori( sequence )
print("Prob: {}".format( prob ))
print("\n".join( state[1].name for state in states ))


# Finally we try and reproduce the transition matrix from 100,000 samples.

# In[12]:


print("Should produce a matrix close to the following: ")
print(" [ [ 0.60, 0.10, 0.30 ] ")
print("   [ 0.40, 0.40, 0.20 ] ")
print("   [ 0.05, 0.15, 0.80 ] ] ")
print()
print("Transition Matrix From 100000 Samples:")
sample, path = model.sample( 100000, path=True )
trans = np.zeros((3,3))

for state, n_state in it.izip( path[1:-2], path[2:-1] ):
	state_name = int( state.name[1:] )-1
	n_state_name = int( n_state.name[1:] )-1
	trans[ state_name, n_state_name ] += 1

trans = (trans.T / trans.sum( axis=1 )).T
print(trans)

