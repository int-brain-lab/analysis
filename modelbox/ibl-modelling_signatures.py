class Dataset:
    """ Object containing data that can be preprocessed.
    Inputs: dataframe # pandas
    Methods: preprocess
    """


class Model:
    """Abstract class for defining models.
    Attributes: 
        parameter_list # list of objects from Parameter class
        name # string
        definition # string
        loglikelihood_function # function handle
    Methods:
        preprocess
        fit
        simulate
    """


class PsychometricFunction(Model):
    """Class for representing psychometric functions. 
        Grab different functions from psignifit, etc
    """


class Parameter:
    """
    Attributes:
        name # string
        description # string
        bounds # float (lower, upper, hard, plausible)
        typical_value # float
        parameterization # string
    """


class FittingMethod:
    """ Abstract class with wrappers to e.g. skikit-learn functions
    Attributes:
        name # string
        algorithm # function handle
    Methods:
        fit
    """


class MaximumLikelihoodEstimation(FittingMethod):
    """ Maximum Likelihood Estimation
    """ 

class PosteriorEstimation(FittingMethod):
    """ Maximum Likelihood Estimation
    """ 


class FittedOutput:
    """ Abstract class for the results of a model fit.
    Name to be agreed on. FittedOutput/FittedResult/?
    Attributes:
        model # dictionary of model identifier (e.g. datajoint primary key of Model)
        data # dictionary of data identifier (e.g. datajoint primary key of DataSet)
        model_metrics # dictionary 
    Methods
        simulate (calls Model.simulate with some parameter set)
        diagnose
        plot
        parameter_recovery
    """


class MaximumLikelihoodOutput(FittedModel):
    """
    Attributes
        starting_points # num_startingpoints x num_params (df?)
        loglikelihoods # num_startingpoints
        maximum_points # num_startingpoints x num_params
    """

class PosteriorOutput(FittedModel):
    """
    Attributes
        posterior
    Methods
        draw_sample
    """

class MCMCPosteriorOutput(PosteriorOutput):
    """
    this class will most likely be a PyMC object
    """
