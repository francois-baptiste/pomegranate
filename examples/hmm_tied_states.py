
# coding: utf-8

# # Tied States Hidden Markov Model

# authors:<br>
# Jacob Schreiber [<a href="sendto:jmchreiber91@gmail.com">jmchreiber91@gmail.com</a>],<br>
# Nicholas Farn [<a href="sendto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# An example of using tied states to represent the same distribution across multiple states. This example is a toy example derived from biology, where we will look at DNA sequences.
# 
# The fake structure we will pretend exists is:
# ```
# start -> background -> CG island -> background -> poly-T region
# ```
# DNA is comprised of four nucleotides, A, C, G, and T. Lets say that in the background sequence, all of these occur at the same frequency. In the CG island, the nucleotides C and G occur more frequently. In the poly T region, T occurs most frequently.
# 
# We need the graph structure, because we fake know that the sequence must return to the background distribution between the CG island and the poly-T region. However, we also fake know that both background distributions need to be the same.

# In[1]:
from pomegranate.base import State
from pomegranate.hmm import HiddenMarkovModel

from pomegranate import *
import random
import numpy as np

from pomegranate.distributions import DiscreteDistribution

random.seed(0)


# Lets start off with an example without tied states and see what happens.

# In[2]:


untiedmodel = HiddenMarkovModel( "No Tied States" )


# Here we'll define the four states.

# In[3]:


background_one = State( DiscreteDistribution({'A': 0.25, 'C':0.25, 'G': 0.25, 'T':0.25 }), name="B1" )
CG_island = State( DiscreteDistribution({'A': 0.1, 'C':0.4, 'G': 0.4, 'T':0.1 }), name="CG" )
background_two = State( DiscreteDistribution({'A': 0.25, 'C':0.25, 'G': 0.25, 'T':0.25 }), name="B2" )
poly_T = State( DiscreteDistribution({'A': 0.1, 'C':0.1, 'G': 0.1, 'T':0.7 }), name="PT" )


# Then add the starting transitions.

# In[4]:


untiedmodel.add_transition( untiedmodel.start, background_one, 1. )


# The transition matrix.

# In[5]:


untiedmodel.add_transition( background_one, background_one, 0.9 )
untiedmodel.add_transition( background_one, CG_island, 0.1 )
untiedmodel.add_transition( CG_island, CG_island, 0.8 )
untiedmodel.add_transition( CG_island, background_two, 0.2 )
untiedmodel.add_transition( background_two, background_two, 0.8 )
untiedmodel.add_transition( background_two, poly_T, 0.2 )
untiedmodel.add_transition( poly_T, poly_T, 0.7 )


# And finally the ending transitions.

# In[6]:


untiedmodel.add_transition( poly_T, untiedmodel.end, 0.3)


# Finishing with the method "bake" to finalize the structure of our model.

# In[7]:


untiedmodel.bake( verbose=True )


# Now let's define the following sequences. Keep in mind training must by done on a list of lists, not on a string in order to allow strings of any length.

# In[8]:


sequences = [ np.array(list("TAGCACATCGCAGCGCATCACGCGCGCTAGCATATAAGCACGATCAGCACGACTGTTTTT")),
			  np.array(list("TAGAATCGCTACATAGACGCGCGCTCGCCGCGCTCGATAAGCTACGAACACGATTTTTTA")),
			  np.array(list("GATAGCTACGACTACGCGACTCACGCGCGCGCTCCGCATCAGACACGAATATAGATAAGATATTTTTT")) ]


# Lets check our distributions before training our model.

# In[9]:


print("\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in untiedmodel.states if not state.is_silent() ))


# Now lets train our model.

# In[10]:


untiedmodel.fit( sequences, stop_threshold=0.01 )


# And check our new distributions after training.

# In[11]:


print("\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in untiedmodel.states if not state.is_silent() ))


# Now we can try our example with tied states.

# In[12]:


tiedmodel = HiddenMarkovModel( "Tied States" )


# Lets redefine the four states.

# In[13]:


background = DiscreteDistribution({'A': 0.25, 'C':0.25, 'G': 0.25, 'T':0.25 })

background_one = State( background, name="B1" )
CG_island = State( DiscreteDistribution({'A': 0.1, 
	'C':0.4, 'G': 0.4, 'T':0.1 }), name="CG" )
background_two = State( background, name="B2" )
poly_T = State( DiscreteDistribution({'A': 0.1, 
	'C':0.1, 'G': 0.1, 'T':0.7 }), name="PT" )


# Then add the starting transitions.

# In[14]:


tiedmodel.add_transition( tiedmodel.start, background_one, 1. );


# Then the tranisiton matrix.

# In[15]:


tiedmodel.add_transition( background_one, background_one, 0.9 )
tiedmodel.add_transition( background_one, CG_island, 0.1 )
tiedmodel.add_transition( CG_island, CG_island, 0.8 )
tiedmodel.add_transition( CG_island, background_two, 0.2 )
tiedmodel.add_transition( background_two, background_two, 0.8 )
tiedmodel.add_transition( background_two, poly_T, 0.2 )
tiedmodel.add_transition( poly_T, poly_T, 0.7 )


# Finally adding the ending transitions.

# In[16]:


tiedmodel.add_transition( poly_T, tiedmodel.end, 0.3 )


# We "bake" the model to finalize its structure.

# In[17]:


tiedmodel.bake( verbose=True )


# Now let's use the following sequences to train our model.

# In[18]:


sequences = [ np.array(list("TAGCACATCGCAGCGCATCACGCGCGCTAGCATATAAGCACGATCAGCACGACTGTTTTT")),
			  np.array(list("TAGAATCGCTACATAGACGCGCGCTCGCCGCGCTCGATAAGCTACGAACACGATTTTTTA")),
			  np.array(list("GATAGCTACGACTACGCGACTCACGCGCGCGCTCCGCATCAGACACGAATATAGATAAGATATTTTTT")) ]


# But before that let's check the distributions in our model.

# In[19]:


print("\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in tiedmodel.states if not state.is_silent() ))


# Now let's train our model.

# In[20]:


tiedmodel.fit( sequences, stop_threshold=0.01 )


# Now let's check our new distributions.

# In[21]:


print("\n".join( "{}: {}".format( state.name, state.distribution ) 
	for state in tiedmodel.states if not state.is_silent() ))


# Notice that states B1 and B2 are the same after training with tied states, not so without tied states.
