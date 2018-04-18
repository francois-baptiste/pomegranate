
# coding: utf-8

# # Hidden Markov Model Example

# authors:<br>
# Jacob Schreiber [<a href="mailto:jmschreiber91@gmail.com">jmschreiber91@gmail.com</a>]<br>
# Nicholas Farn [<a href="mailto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# A simple example highlighting how to build a model using states, add
# transitions, and then run the algorithms, including showing how training
# on a sequence improves the probability of the sequence.

# In[1]:


import random

from pomegranate.base import State
from pomegranate.distributions import UniformDistribution, NormalDistribution
from pomegranate.hmm import HiddenMarkovModel

from pomegranate import *

random.seed(0)


# First we will create the states of the model, one uniform and one normal.

# In[2]:


state1 = State( UniformDistribution(0.0, 1.0), name="uniform" )
state2 = State( NormalDistribution(0, 2), name="normal" )


# We will then create the model by creating a HiddenMarkovModel instance. Then we will add the states.

# In[3]:


model = HiddenMarkovModel( name="ExampleModel" )
model.add_state( state1 )
model.add_state( state2 )


# Now we'll add the start states to the model.

# In[4]:


model.add_transition( model.start, state1, 0.5 )
model.add_transition( model.start, state2, 0.5 )


# And the transition matrix.

# In[5]:


model.add_transition( state1, state1, 0.4 )
model.add_transition( state1, state2, 0.4 )
model.add_transition( state2, state2, 0.4 )
model.add_transition( state2, state1, 0.4 )


# Finally the ending states to the model.

# In[6]:


model.add_transition( state1, model.end, 0.2 )
model.add_transition( state2, model.end, 0.2 )


# To finalize the model, we "bake" it.

# In[7]:


model.bake()


# New we'll create a sample sequence using our model.

# In[8]:


sequence = model.sample()
print(sequence)


# Now we'll feed the sequence through a forward algorithm with our model.

# In[9]:


print(model.forward( sequence )[ len(sequence), model.end_index ])


# Next we'll do the same, except with a backwards algorithm.

# In[10]:


print(model.backward( sequence )[0,model.start_index])


# Then we'll feed the sequence again, through a forward-backward algorithm.

# In[11]:


trans, ems = model.forward_backward( sequence )
print(trans)
print(ems)


# Finally we'll train our model with our example sequence.

# In[12]:


model.fit( [ sequence ] )


# Then repeat the algorithms we fed the sequence through before on our improved model.

# In[13]:


print("Forward")
print(model.forward( sequence )[ len(sequence), model.end_index ])
print()
print("Backward")
print(model.backward( sequence )[ 0,model.start_index ])
print()
trans, ems = model.forward_backward( sequence )
print(trans)
print(ems)

