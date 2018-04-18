
# coding: utf-8

# # Distribution Test Tables

# This example demonstrates how to create some conditional probability tables and a bayesian network.

# In[8]:
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.base import State

from pomegranate import *
import math


# First let's define some conditional probability tables.

# In[9]:
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable

c_table = [[0, 0, 0, 0.6],
		   [0, 0, 1, 0.4],
		   [0, 1, 0, 0.7],
		   [0, 1, 1, 0.3],
		   [1, 0, 0, 0.2],
		   [1, 0, 1, 0.8],
		   [1, 1, 0, 0.9],
		   [1, 1, 1, 0.1]]

d_table = [[ 0, 0, 0.5 ],
		   [ 0, 1, 0.5 ],
		   [ 1, 0, 0.3 ],
		   [ 1, 1, 0.7 ]]

f_table = [[ 0, 0, 0, 0.8 ],
		   [ 0, 0, 1, 0.2 ],
		   [ 0, 1, 0, 0.3 ],
		   [ 0, 1, 1, 0.7 ],
		   [ 1, 0, 0, 0.6 ],
		   [ 1, 0, 1, 0.4 ],
		   [ 1, 1, 0, 0.9 ],
		   [ 1, 1, 1, 0.1 ]]

e_table = [[ 0, 0, 0.7 ],
		   [ 0, 1, 0.3 ],
		   [ 1, 0, 0.2 ],
		   [ 1, 1, 0.8 ]]

g_table = [[ 0, 0, 0, 0.34 ],
		   [ 0, 0, 1, 0.66 ],
		   [ 0, 1, 0, 0.83 ],
		   [ 0, 1, 1, 0.17 ],
		   [ 1, 0, 0, 0.77 ],
		   [ 1, 0, 1, 0.23 ],
		   [ 1, 1, 0, 0.12 ],
		   [ 1, 1, 1, 0.88 ]]


# Then let's convert them into distribution objects.

# In[10]:


a = DiscreteDistribution({ 0: 0.5, 1: 0.5 })
b = DiscreteDistribution({ 0: 0.7, 1: 0.3 })
e = ConditionalProbabilityTable( e_table, [b] )
c = ConditionalProbabilityTable( c_table, [a,b] )
d = ConditionalProbabilityTable( d_table, [c] )
f = ConditionalProbabilityTable( f_table, [c,e] )
g = ConditionalProbabilityTable( g_table, [c,e] )


# Next we can convert these distributions into states.

# In[11]:


a_s = State( a, "a" )
b_s = State( b, "b" )
c_s = State( c, "c" )
d_s = State( d, "d" )
e_s = State( e, "e" )
f_s = State( f, "f" )
g_s = State( g, "g" )


# Now that we have our states created, we can finally start making our bayesian network.

# In[12]:


model = BayesianNetwork( "derp" )
model.add_nodes( a_s, b_s, c_s, d_s, e_s, f_s, g_s )


# Then we define the edges.

# In[13]:


model.add_edge( a_s, c_s )
model.add_edge( b_s, c_s )
model.add_edge( c_s, d_s )
model.add_edge( c_s, f_s )
model.add_edge( b_s, e_s )
model.add_edge( e_s, f_s )
model.add_edge( c_s, g_s )
model.add_edge( e_s, g_s )


# We finish by baking the network to finalize its structure.

# In[14]:


model.bake()


# Now we can check on the structure of our bayesian network.

# In[15]:


print("\n".join( "{:10.10} : {}".format( state.name, belief.parameters[0] ) for state, belief in zip( model.states, model.predict_proba( max_iterations=100 ) ) ))

