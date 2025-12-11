from __future__ import division

import copy
import os
import pickle
import numpy as np
from scipy.special import logsumexp
import scipy.constants

from . import utils
from . import utils_lbol

ln10 = np.log(10)

model_parameters_dict = {
    "Arnett": ["tau_m", "log10_mni"],
    "Arnett_modified": ["tau_m", "log10_mni", "t_0"],
}


class SimpleBolometricLightCurveModel(object):
    def __init__(self, sample_times, parameter_conversion=None, model="Arnett"):
        """A light curve model object

        An object to evaluted the kilonova (with Me2017) light curve across filters

        Parameters
        ----------
        sample_times: np.array
            An arry of time for the light curve to be evaluted on

        Returns
        -------
        LightCurveModel: `nmma.em.model.SimpleBolometricLightCurveModel`
            A light curve model object, able to evaluted the light curve
            give a set of parameters
        """

        assert model in model_parameters_dict.keys(), (
            "Unknown model," "please update model_parameters_dict at em/model.py"
        )

        self.model = model
        self.model_parameters = model_parameters_dict[model]
        self.sample_times = sample_times
        self.parameter_conversion = parameter_conversion

    def __repr__(self):
        return self.__class__.__name__ + "(model={0})".format(self.model)

    def generate_lightcurve(self, sample_times, parameters):

        if self.parameter_conversion:
            new_parameters = parameters.copy()
            new_parameters, _ = self.parameter_conversion(new_parameters, [])
        else:
            new_parameters = parameters.copy()

        if self.model == "Arnett":
            lbol = utils_lbol.arnett_lc(sample_times, new_parameters)
        elif self.model == "Arnett_modified":
            lbol = utils_lbol.arnett_modified_lc(sample_times, new_parameters)

        return lbol
