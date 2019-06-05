""" Bayesian model of confirmation bias
Anne Urai, CSHL, 2019
urai@cshl.edu

"""

import pandas as pd
import numpy as np
import sys, os, time
import seaborn as sns
from IPython import embed as shell

# ============================= #
# ============================= #

# 1. stimulus distribution, across trials
svec = np.arange(-10, 10, 1000)

# shuffle their order
stimuli = svec.shuffle()

# 2. prior, start of first trial
# ASSUME A UNIFORM PRIOR THAT REFLECTS THE STIMULUS?
prior = np.normpdf(svec, 0, 2)

# some level of sensory noise
sensorynoise = 1

# 3. on each trial, draw a sensory observation
for i, stim in enumerate(stimuli):

	# draw a sensory likelihood distribution - independent of the prior
	likelihood = normpdf(svec, stim, sensorynoise)

	# combine the two
	proto_posterior = likelihood .* prior

	# normalize to get posterior
	posterior = proto_posterior .* sum(proto_posterior);

	# make a choice
	choice[i] = np.sign(posterior.max())

	# IMPORTANT: USE THIS POSTERIOR FOR THE NEXT PRIOR!
	prior = posterior

	# ADD IN LUU & STOCKER CUT-OFF POSTERIOR
	prior = posterior(np.sign(posterior) == np.sign(choice))

# 4a. do these agents have serial choice bias?

# 4b. does the serial choice bias scale with confidence?


#TODO: 1. add within-trial integration and fit this with the DDM
#TODO: 2. add a specific learning rule (possibly feedback-dependent rather than simply cutting off the posterior)