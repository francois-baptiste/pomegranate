
# coding: utf-8

# # Bayesian Networks in Naive Bayes

# author: Nicholas Farn [<a href="sendto:nicholasfarn@gmail.com">nichoalsfarn@gmail.com</a>]

# In this example we will compare two bayesian networks and see which fits a set of data better through the use of a Naive Bayes classifier. In this example we will see if a set of data corresponds more to a city in an arid or wet climate.
# 
# This will be done by checking how often the grass is wet, whether it is due to rain or sprinklers. This is a variation of the [example](https://en.wikipedia.org/wiki/Bayesian_network#Example) given by Wikipedia on Bayesian Networks.

# In[1]:
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.NaiveBayes import NaiveBayes
from pomegranate.base import State

from pomegranate import *
import numpy as np


# Let's create the bayesian network for our aridcity first. You can think of an arid city as a place like Pheonix, Arizona.

# In[2]:
from pomegranate.distributions import DiscreteDistribution, ConditionalProbabilityTable

arid_rain = DiscreteDistribution( { 'T': 0.1, 'F': 0.9 } )


# The probability that people will use a sprinkler is dependent upon the probability that it has rained.

# In[3]:


arid_sprinkler = ConditionalProbabilityTable(
    [[ 'T', 'T', 0.01 ],
     [ 'T', 'F', 0.99 ],
     [ 'F', 'T', 0.5 ],
     [ 'F', 'F', 0.6 ]], [arid_rain])


# Finally there is the probability that the grass is wet, which is dependent upon both the rain and whether the sprinklers are on.

# In[4]:


arid_grass = ConditionalProbabilityTable(
    [[ 'T', 'T', 'T', 0.99 ],
     [ 'T', 'T', 'F', 0.01 ],
     [ 'T', 'F', 'T', 0.8 ],
     [ 'T', 'F', 'F', 0.3 ],
     [ 'F', 'T', 'T', 0.9 ],
     [ 'F', 'T', 'F', 0.1 ],
     [ 'F', 'F', 'T', 0.0 ],
     [ 'F', 'F', 'F', 1.0 ]], [ arid_rain, arid_sprinkler ])


# Next we need to create the states for our "arid" bayesian network.

# In[5]:


s1 = State( arid_rain, name="rain" )
s2 = State( arid_sprinkler, name="sprinkler" )
s3 = State( arid_grass, name="grass" )


# Then the bayesian network itself, along with its states and transitions between them.

# In[6]:


arid = BayesianNetwork( "arid" )
arid.add_states(  s1, s2, s3  )
arid.add_transition( s1, s2 )
arid.add_transition( s1, s3 )
arid.add_transition( s2, s3 )


# Finally we need to "bake" our bayesian network to finalize its structure.

# In[7]:


arid.bake()


# Now that we've created a bayesian network for an arid city, we can create our bayesian network for our wet city. We can think of a wet city as a place like Seattle, Washington, which has a high probability of rainfall.
# 
# The bayesian network is the same as before, just with different probabilities. So we will start with similar probability distributions.

# In[8]:


wet_rain = DiscreteDistribution( { 'T': 0.6, 'F': 0.4 } )

wet_sprinkler = ConditionalProbabilityTable(
    [[ 'T', 'T', 0.01 ],
     [ 'T', 'F', 0.99 ],
     [ 'F', 'T', 0.4 ],
     [ 'T', 'F', 0.6 ]], [ wet_rain ] )

wet_grass = ConditionalProbabilityTable(
    [[ 'T', 'T', 'T', 0.99 ],
     [ 'T', 'T', 'F', 0.01 ],
     [ 'T', 'F', 'T', 0.8 ],
     [ 'T', 'F', 'F', 0.2 ],
     [ 'F', 'T', 'T', 0.9 ],
     [ 'F', 'T', 'F', 0.1 ],
     [ 'F', 'F', 'T', 0.0 ],
     [ 'F', 'F', 'F', 1.0 ]], [ wet_rain, wet_sprinkler ])


# Then our states

# In[9]:


s4 = State( wet_rain, "rain" )
s5 = State( wet_sprinkler, "sprinkler" )
s6 = State( wet_grass, "grass" )


# Then create our bayesian network.

# In[10]:


wet = BayesianNetwork( "wet" )
wet.add_states(  s4, s5, s6  )
wet.add_transition( s4, s5 )
wet.add_transition( s4, s6 )
wet.add_transition( s5, s6 )


# And "bake" it to finalize its structure.

# In[11]:


wet.bake()


# Now that we have finally created our two bayesian networks, we can finally make our naive bayes classifier.

# In[12]:


clf = NaiveBayes( [ arid, wet ] )


# Something to note is that while both bayesian networks have the same structure in this example, that is not necessary to use bayesian networks for naive bayes. All that is needed is for them to share the same inputs.

# We can now produce our data set to test out our naive bayes classifier. Currently, in order to recieve a probability, all states in a bayesian network have to be specified.

# In[13]:


data = np.array( [[ 'T', 'F', 'T' ],
                  [ 'T', 'F', 'T' ],
                  [ 'T', 'F', 'F' ],
                  [ 'F', 'T', 'T' ],
                  [ 'T', 'T', 'T' ]] )


# Now we can check the probabilities of each sample occurring under each bayesian network.

# In[14]:


clf.predict_proba( data )


# And the classification of the data overall.

# In[15]:


clf.predict( data )


# As we can see, more data seems to occur under the second bayesian network. Therefore the data must occur more often for a city in a "wet" enviroment.
