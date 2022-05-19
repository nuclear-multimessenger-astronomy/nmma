import os
import copy
import glob
import numpy as np

from ..em import training, utils, model_parameters


def test_training():

    # The number of PCA components we'll use to represent each lightcurve
    n_coeff = 3
    model_name = "Bu2019lm_sparse"

    # The array of times we'll use to examine each lightcurve
    tini, tmax, dt = 0.1, 5.0, 0.2
    tt = np.arange(tini, tmax + dt, dt)  #

    # The filters we'll be focusing on
    filts = [
        "g",
        "r",
    ]  # We will focus on these two bands; all available: ["u","g","r","i","z","y","J","H","K"]

    dataDir = f"{os.path.dirname(__file__)}/data/bulla"
    ModelPath = "svdmodels"
    filenames = glob.glob("%s/*.dat" % dataDir)

    data = utils.read_photometry_files(filenames, filters=filts)
    # Loads the model data
    training_data, parameters = model_parameters.Bu2019lm_sparse(data)

    interpolation_type = "sklearn_gp"
    training.SVDTrainingModel(
        model_name,
        copy.deepcopy(training_data),
        parameters,
        tt,
        filts,
        n_coeff=n_coeff,
        svd_path=ModelPath,
        interpolation_type=interpolation_type,
    )

    interpolation_type = "tensorflow"
    training.SVDTrainingModel(
        model_name,
        copy.deepcopy(training_data),
        parameters,
        tt,
        filts,
        n_coeff=n_coeff,
        svd_path=ModelPath,
        interpolation_type=interpolation_type,
    )
