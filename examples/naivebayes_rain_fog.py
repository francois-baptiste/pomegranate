
# coding: utf-8

# # Rain Influences Fog or Vice Versa?

# author: Nicholas Farn [<a href ="sendto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# In this example we will use a naive bayes classifier to compare two models for a bayesian network, one in which rain influences the probability of fog occurring, another in which fog influences the probability of rain occurring, and a third in which they are independent from one another. All of which affect whether or not grass is wet.

# In[39]:
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.NaiveBayes import NaiveBayes
from pomegranate.base import State

from pomegranate import *
import numpy as np


# First we'll create the bayesian network in which rain influences the occurrence of fog. To do that we first create a discrete distribution for the occurrence of rain.

# In[40]:
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable

dist_rain = DiscreteDistribution( { 'T': 0.6, 'F': 0.4 } )


# Now we can create a conditional probability table for fog which is dependent on rain.

# In[41]:


dist_fog = ConditionalProbabilityTable(
    [[ 'T', 'T', 0.1 ],
     [ 'T', 'F', 0.9 ],
     [ 'F', 'T', 0.4 ],
     [ 'F', 'F', 0.6 ]], [ dist_rain ] )


# And finally the conditional probability table for whether the grass is wet, which is dependent upon rain and fog.

# In[42]:


dist_grass = ConditionalProbabilityTable(
    [[ 'T', 'T', 'T', 0.99 ],
     [ 'T', 'T', 'F', 0.01 ],
     [ 'T', 'F', 'T', 0.9 ],
     [ 'T', 'F', 'F', 0.1 ],
     [ 'F', 'T', 'T', 0.7 ],
     [ 'F', 'T', 'F', 0.3 ],
     [ 'F', 'F', 'T', 0.1 ],
     [ 'F', 'F', 'F', 0.9 ]], [ dist_rain, dist_fog ] )


# Now that we have our distributions, we can create states for our bayesian network out of them.

# In[43]:


rain = State( dist_rain, 'rain' )
fog = State( dist_fog, 'fog' )
grass = State( dist_grass, 'grass' )


# And finally we can create our bayesian network in which fog is dependent upon rain. We add the states in and the possible transitions as well.

# In[44]:


rain2fog = BayesianNetwork( 'rain2fog' )
rain2fog.add_states(  rain, fog, grass  )
rain2fog.add_transition( rain, fog )
rain2fog.add_transition( rain, grass )
rain2fog.add_transition( fog, grass )


# Finally we bake it in order to finalize it's structure.

# In[45]:


rain2fog.bake()


# Now let's create our bayesian network in which fog influencesthe occurrence of rain. This process is similar to the creation of our last bayesian network. However instead this time we start by creating a discrete distribution for the occurrence of fog.

# In[46]:


dist_fog = DiscreteDistribution( { 'T': 0.7, 'F': 0.3 } )


# Then the conditional probability table for rain, which is logically dependent on fog.

# In[47]:


dist_rain = ConditionalProbabilityTable(
    [[ 'T', 'T', 0.2 ],
     [ 'T', 'F', 0.8 ],
     [ 'F', 'T', 0.6 ],
     [ 'F', 'F', 0.4 ]], [ dist_fog ])


# And the conditional probability table for whether the grass is wet, which is dependent on fog and rain.

# In[48]:


dist_grass = ConditionalProbabilityTable(
    [[ 'T', 'T', 'T', 0.99 ],
     [ 'T', 'T', 'F', 0.01 ],
     [ 'T', 'F', 'T', 0.9 ],
     [ 'T', 'F', 'F', 0.1 ],
     [ 'F', 'T', 'T', 0.7 ],
     [ 'F', 'T', 'F', 0.3 ],
     [ 'F', 'F', 'T', 0.1 ],
     [ 'F', 'F', 'F', 0.9 ]], [ dist_rain, dist_fog ] )


# Like last time we convert these distributions into states for our bayesian network.

# In[49]:


rain = State( dist_rain, 'rain' )
fog = State( dist_fog, 'fog' )
grass = State( dist_grass, 'grass' )


# Then create our bayesian network, adding in our states and possible transitions.

# In[50]:


fog2rain = BayesianNetwork( 'fog2rain' )
fog2rain.add_states(  rain, fog, grass  )
fog2rain.add_transition( fog, rain )
fog2rain.add_transition( fog, grass )
fog2rain.add_transition( rain, grass )


# Then we bake it to finalize its structure.

# In[51]:


fog2rain.bake()


# Then finally there's our bayesian network in which the occurrence of rain and fog are independent of one another. We start creating this by making discrete distributions for rain and fog.

# In[52]:


dist_rain = DiscreteDistribution( { 'T': 0.4, 'F': 0.7 } )
dist_fog = DiscreteDistribution( { 'T': 0.5, 'F': 0.5 } )


# Then the conditional probability table for whether the grass is wet.

# In[53]:


dist_grass = ConditionalProbabilityTable(
	[[ 'T', 'T', 'T', 0.99 ],
	 [ 'T', 'T', 'F', 0.01 ],
	 [ 'T', 'F', 'T', 0.9 ],
	 [ 'T', 'F', 'F', 0.1 ],
	 [ 'F', 'T', 'T', 0.7 ],
	 [ 'F', 'T', 'F', 0.3 ],
	 [ 'F', 'F', 'T', 0.1 ],
	 [ 'F', 'F', 'F', 0.9 ]], [ dist_rain, dist_fog ] )


# Convert these into states.

# In[54]:


rain = State( dist_rain, 'rain' )
fog = State( dist_fog, 'fog' )
grass = State( dist_grass, 'grass' )


# Then create our bayesian network.

# In[55]:


indie = BayesianNetwork( 'indie' )
indie.add_states( rain, fog, grass  )
indie.add_transition( rain, grass )
indie.add_transition( fog, grass )


# Then finishing by calling the method bake to finalize its structure.

# In[56]:


indie.bake()


# Now we can finally create our naive bayes classifier. Compared to before, this is relatively straight forward.

# In[60]:


clf = NaiveBayes( [ rain2fog, fog2rain, indie ] )


# Done! Now let's test it out on the following set of data. Note that each state must be specified in the input.

# In[61]:


data = np.array( [[ 'T', 'F', 'T' ],
                  [ 'T', 'F', 'T' ],
                  [ 'T', 'F', 'F' ],
                  [ 'F', 'T', 'T' ],
                  [ 'T', 'T', 'T' ]] )


# First we can check the probabilities of each sample occurring under each bayesian network.

# In[62]:


clf.predict_proba( data )


# And the classification of the data overall.

# In[ ]:


clf.predict( data )


# It looks like our first bayesian network fits the data better overall, though not be much.

# We can also train our naive bayes classifier to fit a set of data. For instance the following.

# In[ ]:


X = np.array([[ 'T', 'F', 'F' ],
              [ 'T', 'T', 'F' ],
              [ 'T', 'T', 'T' ],
              [ 'F', 'F', 'F' ],
              [ 'T', 'F', 'F' ],
              [ 'F', 'T', 'F' ],
              [ 'T', 'T', 'F' ],
              [ 'T', 'F', 'T' ],
              [ 'T', 'F', 'T' ],
              [ 'F', 'T', 'T' ],
              [ 'T', 'F', 'T' ],
              [ 'F', 'T', 'T' ],
              [ 'F', 'F', 'T' ],
              [ 'F', 'T', 'T' ],
              [ 'F', 'F', 'T' ]])

y = np.array([ 2, 2, 2, 0, 2, 1, 0, 0, 2, 2, 0, 2, 1, 1, 2 ])


# This is also rather straightforward to do. Just make sure that every state is specified for the sample and that both input arrays have the same length.

# In[ ]:


clf.fit( X, y )


# Using the same data as before we can check the new probabilities.

# In[ ]:


clf.predict_proba( data )


# And classifications.

# In[ ]:


clf.predict( data )


# Looks like the second bayesian network fits the data better this time, however by only a slim margin.
