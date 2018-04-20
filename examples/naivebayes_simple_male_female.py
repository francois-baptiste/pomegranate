
# coding: utf-8

# # Naive Bayes Simple Male or Female

# author: Nicholas Farn [<a href="sendto:nicholasfarn@gmail.com">nicholasfarn@gmail.com</a>]

# This example shows how to create a simple Gaussian Naive Bayes Classifier using pomegranate. In this example we will be given a set of data measuring a person's height (feet) and try to classify them as male or female. This example is a simplification drawn from the example in the Wikipedia <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples">article</a> on Naive Bayes Classifiers.

# In[1]:
from pomegranate.NaiveBayes import NaiveBayes
from pomegranate.distributions import NormalDistribution

from pomegranate import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')


# First we'll create the distributions for our model. In this case we'll assume that height, weight, and foot size are normally distributed. We'll fit our distribution to a set of data for males and females.

# In[2]:


male = NormalDistribution.from_samples([182.88  , 180.4416, 170.0784, 180.4416, 185.3184, 177.6984])
female = NormalDistribution.from_samples([152.4   , 167.64  , 165.2016, 175.26  , 157.5816, 152.4   ])


# Let's check on the parameters for our male and female height distributions.

# In[3]:

plt.close("all")

male.plot( n=100000, edgecolor='c', color='c', bins=50, label='Male' )
female.plot( n=100000, edgecolor='g', color='g', bins=50, label='Female' )
plt.legend( fontsize=14 )
plt.ylabel('Count')
plt.xlabel('Height (ft)')
plt.show()

print("Male distribution has mu = {:.3} and sigma = {:.3}".format( *male.parameters ))
print("Female distribution has mu = {:.3} and sigma = {:.3}".format( *female.parameters ))


# Everything seems to look good so let's create our Naive Bayes Classifier.

# In[4]:


clf = NaiveBayes([ male, female ])


# Let's take a look at how our classifier calls people of various heights. We can either look at a probabilistic measurement of the sample being male or female, or a hard call prediction. Lets take a look at both.

# In[5]:


data = np.array([ 152   , 182  , 150, 173 ])

for sample in data:
    probability = clf.predict_proba(sample)
    print("Height {:5.3f}, {:5.3f}% chance male and {:5.3f}% chance female".format( sample, 100*probability[0,0], 100*probability[0,1]))


# In[6]:

data = np.array([ 152   , 182  , 150, 168 ])

for sample in data:
    result = clf.predict(sample)[0]
    print("Person with height {} is {}.".format( sample, "female" if result else "male" ))


# These results look good. We can also train a our classifier with a set of data. This is done by creating a set of observations along with a set with the corresponding correct classification.

# In[7]:

X = np.array([ [90], [95], [85], [82.5], [50], [75], [65], [75] ])
y = np.array([ 0, 0, 0, 0, 1, 1, 1, 1 ])

clf.fit( X, y )


# In this case we fitted the normal distributions to fit a set of data with male an female weights (lbs). Let's check the results with the following data set.

# In[8]:


data = np.array([ 65. , 100. ,  50. ,  81. ,  72.5])


# Now let's enter it into our classifier.

# In[9]:

for sample in data:
    result = clf.predict(sample)[0]
    print("Person with weight {} is {}.".format( sample, "female" if result else "male" ))


# Everything looks good from here. In this tutorial we created a simple Naive Bayes Classifier with normal distributions. It is possible to create a classifier with more complex distributions, or even with a Hidden Markov Model.
