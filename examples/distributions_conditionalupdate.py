
# coding: utf-8

# # Conditional Distributions Update

# authors:<br>
# Jacob Schreiber [<a href="mailto:jmschreiber91@gmail.com">jmschreiber91@gmail.com</a>]<br>
# Nicholas Farn [<a href="mailto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# This example shows the implementation of the classic <a href="https://en.wikipedia.org/wiki/Monty_Hall">Monty Hall</a> problem.

# In[1]:


from pomegranate import *
import numpy as np


# Lets create the distributions for the guest's choice and prize's location. They are both discrete distributions and are independent of one another.

# In[2]:
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable

guest = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
prize = DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )


# Now we'll create a conditional probability table for the Monty Hall problem. The results of the Monty Hall problem is dependent on both the guest and the prize.

# In[3]:


monty = ConditionalProbabilityTable(
	[[ 'A', 'A', 'A', 0.0 ],
	 [ 'A', 'A', 'B', 0.5 ],
	 [ 'A', 'A', 'C', 0.5 ],
	 [ 'A', 'B', 'A', 0.0 ],
	 [ 'A', 'B', 'B', 0.0 ],
	 [ 'A', 'B', 'C', 1.0 ],
	 [ 'A', 'C', 'A', 0.0 ],
	 [ 'A', 'C', 'B', 1.0 ],
	 [ 'A', 'C', 'C', 0.0 ],
	 [ 'B', 'A', 'A', 0.0 ],
	 [ 'B', 'A', 'B', 0.0 ],
	 [ 'B', 'A', 'C', 1.0 ],
	 [ 'B', 'B', 'A', 0.5 ],
	 [ 'B', 'B', 'B', 0.0 ],
	 [ 'B', 'B', 'C', 0.5 ],
	 [ 'B', 'C', 'A', 1.0 ],
	 [ 'B', 'C', 'B', 0.0 ],
	 [ 'B', 'C', 'C', 0.0 ],
	 [ 'C', 'A', 'A', 0.0 ],
	 [ 'C', 'A', 'B', 1.0 ],
	 [ 'C', 'A', 'C', 0.0 ],
	 [ 'C', 'B', 'A', 1.0 ],
	 [ 'C', 'B', 'B', 0.0 ],
	 [ 'C', 'B', 'C', 0.0 ],
	 [ 'C', 'C', 'A', 0.5 ],
	 [ 'C', 'C', 'B', 0.5 ],
	 [ 'C', 'C', 'C', 0.0 ]], [guest, prize] )


# Let's create some sample data to train our model.

# In[4]:


data = [[ 'A', 'A', 'C' ],
		[ 'A', 'A', 'C' ],
		[ 'A', 'A', 'B' ],
		[ 'A', 'A', 'A' ],
		[ 'A', 'A', 'C' ],
		[ 'B', 'B', 'B' ],
		[ 'B', 'B', 'C' ],
		[ 'C', 'C', 'A' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'C', 'C' ],
		[ 'C', 'B', 'A' ]]


# Then train our model and see the results.

# In[5]:


monty.fit( data, weights=[1, 1, 3, 3, 1, 1, 3, 7, 1, 1, 1, 1] )
print(monty)

