
# coding: utf-8

# # Rainy or Sunny Hidden Markov Model

# authors:<br>
# Jacob Schreiber [<a href="mailto:jmschreiber91@gmail.com">jmschreiber91@gmail.com</a>]<br>
# Nicholas Farn [<a href="mailto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# This is an example of a sunny-rainy hidden markov model using yahmm. The example is drawn from the Wikipedia <a href=https://en.wikipedia.org/wiki/Hidden_Markov_model#A_concrete_example>article</a> on Hidden Markov Models describing what Bob likes to do on rainy or sunny days.

# In[1]:
from pomegranate.base import State
from pomegranate.hmm import HiddenMarkovModel

from pomegranate import *
import random
import math

from pomegranate.distributions import DiscreteDistribution

random.seed(0)


# We first create a `HiddenMarkovModel` object, and name it "Rainy-Sunny".

# In[2]:


model = HiddenMarkovModel( name="Rainy-Sunny" )


# We then create the two possible states of the model, "rainy" and "sunny". We make them both discrete distributions, with the possibilities of Bob either walking, shopping, or cleaning.

# In[3]:


rainy = State( DiscreteDistribution({ 'walk': 0.1, 'shop': 0.4, 'clean': 0.5 }), name='Rainy' )
sunny = State( DiscreteDistribution({ 'walk': 0.6, 'shop': 0.3, 'clean': 0.1 }), name='Sunny' )


# We then add the transitions probabilities, starting with the probability the model starts as sunny or rainy.

# In[4]:


model.add_transition( model.start, rainy, 0.6 )
model.add_transition( model.start, sunny, 0.4 )


# We then add the transition matrix. We make sure to subtract 0.05 from each probability to add to the probability of exiting the hmm.

# In[5]:


model.add_transition( rainy, rainy, 0.65 )
model.add_transition( rainy, sunny, 0.25 )
model.add_transition( sunny, rainy, 0.35 )
model.add_transition( sunny, sunny, 0.55 )


# Last, we add transitions to mark the end of the model.

# In[6]:


model.add_transition( rainy, model.end, 0.1 )
model.add_transition( sunny, model.end, 0.1 )


# Finally we "bake" the model, finalizing its structure.

# In[7]:


model.bake( verbose=True )


# Now lets check on Bob each hour and see what he is doing! In other words lets create a sequence of observations.

# In[8]:


sequence = [ 'walk', 'shop', 'clean', 'clean', 'clean', 'walk', 'clean' ]


# Now lets check the probability of observing this sequence.

# In[13]:


print(math.e**model.forward( sequence )[ len(sequence), model.end_index ])


# Then the probability that Bob will be cleaning a step 3 in this sequence.

# In[11]:


print(math.e**model.forward_backward( sequence )[1][ 2, model.states.index( rainy ) ])


# The probability of the sequence occurring given it is Sunny at step 4 in the sequence.

# In[12]:


print(math.e**model.backward( sequence )[ 3, model.states.index( sunny ) ])


# Finally the probable series of states given the above sequence.

# In[15]:


print(" ".join( state.name for i, state in model.maximum_a_posteriori( sequence )[1] ))

