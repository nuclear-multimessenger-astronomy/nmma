import os
import copy
import glob
import numpy as np
import pytest
import shutil

from ..em import training, model_parameters, io
@pytest.fixture(autouse=True)
def cleanup_outdir():
    ModelPath = "svdtrainingmodel"
    yield
    if os.path.exists(ModelPath):
        shutil.rmtree(ModelPath)

def test_training():

    # The number of PCA components we'll use to represent each lightcurve
    n_coeff = 3
    model_name = "Bu2019lm_sparse"

    # The array of times we'll use to examine each lightcurve
    tini, tmax, dt = 0.1, 5.0, 0.2
    tt = np.arange(tini, tmax + dt, dt)  #

    # The filters we'll be focusing on
    filts = [
        "ztfg",
        "ztfr",
    ]  # We will focus on these two bands; all available: ["sdssu","ztfg","ztfr","ztfi","ps1__z","ps1__y","2massj","2massh","2massks"]

    workingDir = os.path.dirname(__file__)

    dataDir = os.path.join(workingDir, "data/bulla")
    ModelPath = "svdtrainingmodel"
    filenames = glob.glob(f"{dataDir}/*.dat")

    data = io.read_photometry_files(filenames, filters=filts)
    # Loads the model data
    training_data, parameters = model_parameters.Bu2019lm_sparse(data)

    interpolation_type = "sklearn_gp"
    training.SklearnGPTrainingModel(
        model_name,
        copy.deepcopy(training_data),
        parameters,
        tt,
        filts,
        n_coeff=n_coeff,
        svd_path=ModelPath,
    )

    training.create_benchmark(
        model_name,
        ModelPath,
        dataDir,
        interpolation_type=interpolation_type,
        filters=filts,
    )

    interpolation_type = "keras"
    training.KerasTrainingModel(
        model_name,
        copy.deepcopy(training_data),
        parameters,
        tt,
        filts,
        n_coeff=n_coeff,
        svd_path=ModelPath,
    )

    training.create_benchmark(
        model_name,
        ModelPath,
        dataDir,
        interpolation_type=interpolation_type,
        filters=filts,
    )
