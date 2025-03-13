import json
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import numpy as np
from .utils import autocomplete_data
from ..utils.models import get_models_home, get_model

try:
    import keras as k
except ImportError:
    print("Install keras and better explicitly set the 'KERAS_BACKEND' \
          as env-variable if you want to use it...")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic

    #NOTE this is used by the keras model!
    from sklearn.model_selection import train_test_split 
except ImportError:
    print("Install scikit-learn if you want to use it...")

try:
    from gp_api.gaussian_process import GaussianProcess
    from gp_api.kernels import CompactKernel
except ImportError:
    print("Install gaussian-process-api if you want to use it...")

class BaseTrainingModel(object):
    """A light curve training model object

    An object to train a light curve model across filters
    by computing a grid-based SVD

    Parameters
    ----------
    model: str
        Name of the model
    data: dict
        Data containing filter data with filters as columns
    parameters: list
        List of model parameters
    sample_times: np.array
        An array of time for the light curve to be evaluated on
    filters: list
        List of filters to train
    svd_path: str, optional
        Path to the svd directory
    n_coeff: int, optional
        Number of eigenvalues to be taken for SVD evaluation. Default is 10
    n_epochs: int, optional
        Number of epochs for model training. Default is 15
    data_type: str, optional
        Data type for interpolation [photometry or spectroscopy]
    data_time_unit: str, optional
        Unit of time for the data [days, hours, minutes, or seconds]. Default is days
    plot: bool, optional
        Whether to show plots or not. Default is False
    plotdir: str, optional
        Directory for plotting
    ncpus: int, optional
        Number of CPUs to use for training
    univariate_spline: bool, optional
        Whether to use univariate spline for interpolation. Default is False
    univariate_spline_s: int, optional
        Smoothing factor for univariate spline
    random_seed: int, optional
        Random seed for reproducibility
    start_training: bool, optional
        Indicate whether we want to start training a model directly after initialization
    continue_training: bool, optional
        Indicate whether we want to continue training an existing model
    """
    def __init__(
        self,
        model,
        data,
        parameters,
        sample_times,
        filters,
        svd_path=None,
        n_coeff=10,
        n_epochs=15,
        data_type="photometry",
        data_time_unit="days",
        plot=False,
        plotdir=os.path.join(os.getcwd(), "plot"),
        ncpus=1,
        univariate_spline=False,
        univariate_spline_s=2,
        random_seed=42,
        start_training=True,
        continue_training=False,
    ):

        self.model = model
        self.svd_path = get_models_home(svd_path)
        self.modelfile = os.path.join(self.svd_path, f"{self.model}.joblib")
        self.outdir = os.path.join(self.svd_path, f"{self.model}{self.model_specifier}")

        self.data = data
        self.model_parameters = parameters
        self.sample_times = sample_times
        self.filters = filters
        self.n_coeff = n_coeff
        self.n_epochs = n_epochs
        self.data_type = data_type
        self.data_time_unit = data_time_unit
        self.plot = plot
        self.plotdir = plotdir
        self.ncpus = ncpus
        self.univariate_spline = univariate_spline
        self.univariate_spline_s = univariate_spline_s
        self.random_seed = random_seed
        if self.univariate_spline:
            print("The grid will be interpolated to sample_time with UnivariateSpline")
        else:
            print("The grid will be interpolated to sample_time with linear interpolation")

        if self.ncpus > 1:
            print(f"Running with {self.ncpus} CPUs")

        self.interpolate_data(data_time_unit=data_time_unit)

        self.model_exists = self.check_model()
        self.start_training = start_training
        self.continue_training = continue_training
        if self.model_exists:
            print("Model exists... will load that model.")
            self.load_model()
        else:
            self.svd_model = self.generate_svd_model()
            if self.continue_training:
                warnings.warn(
                    "Warning: --continue-training set, but no existing model found."
                )

        if (not self.model_exists and self.start_training) or (
            self.model_exists and self.continue_training
        ):
            print("Training model...")
            # self.svd_model = self.generate_svd_model()
            self.train_model()
            self.save_model()

        self.load_model()


    def interpolate_data(self, data_time_unit="days"):
        # Before interpolation, convert input data time values to days
        if data_time_unit in ["days", "day", "d"]:
            time_scale_factor = 1.0
        elif data_time_unit in ["hours", "hour", "hr", "h"]:
            time_scale_factor = 24.0
        elif data_time_unit in ["minutes", "minute", "min", "m"]:
            time_scale_factor = 1440.0
        elif data_time_unit in ["seconds", "second", "sec", "s"]:
            time_scale_factor = 86400.0
        else:
            raise ValueError(
                "data_time_unit must be one of days, hours, minutes, or seconds."
            )
        if self.univariate_spline:
            extension_mode = "spline"
            ref_value = self.univariate_spline_s
        else:
            extension_mode = "linear"
            ref_value = np.nan

        for key in self.data.keys():
            # initialise data array for all filters and sample times
            ##FIXME should better use nans!
            self.data[key]["data"] = np.zeros(
                (len(self.sample_times), len(self.filters))
                )
            obs_times = self.data[key]["t"]/ time_scale_factor

            # Interpolate data onto grid
            if self.data_type == "photometry":
                for j, filt in enumerate(self.filters):
                    self.data[key]['data'][:, j] = autocomplete_data(
                        self.sample_times, obs_times, self.data[key][filt], extrapolate=extension_mode, ref_value=ref_value)
                    del self.data[key][filt]

            elif self.data_type == "spectroscopy":
                for j, filt in enumerate(self.filters):
                    ref_data = np.log10(self.data[key]["fnu"][:, j])
                    log_maginterp = autocomplete_data(self.sample_times,
                        obs_times, ref_data )
                    self.data[key]["data"][:, j] = 10** log_maginterp
                del self.data[key]["fnu"]

                
            del self.data[key]["t"]

    def generate_svd_model(self) -> dict:
        """
        Function that preprocesses the data and performs the SVD decomposition for each filter of the data.
        The SVD is done with np.linalg

        Returns:
            svd_model: A dictionary with keys being the filters. The values are dictionaries containing
            the processed values of the parameters, processed values of the data (normalized into a [0, 1] range)
            and the matrices which are used in the projection of the SVD.
        """

        # Place the relevant parameters into an array
        param_array = []
        for key in self.data.keys():
            param_array.append(
                [self.data[key][param] for param in self.model_parameters]
            )

        param_array_postprocess, param_mins, param_maxs = min_max_scaling(param_array)

        # Output is data with keys being filters, and values being dictionaries for the SVD decomposition
        svd_model = {}
        # Loop through filters
        for jj, filt in enumerate(self.filters):
            print("Normalizing mag filter %s..." % filt)
            data_array = [
                self.data[key]["data"][:, jj] for key in self.data.keys()
            ] 
            data_array_postprocess, mins, maxs = min_max_scaling(data_array)
            data_array_postprocess = np.nan_to_num(data_array_postprocess, nan=0.0)

            svd_model[filt] = {}
            svd_model[filt]["param_array_postprocess"] = param_array_postprocess
            svd_model[filt]["param_mins"] = param_mins
            svd_model[filt]["param_maxs"] = param_maxs
            svd_model[filt]["mins"] = mins
            svd_model[filt]["maxs"] = maxs
            svd_model[filt]["tt"] = self.sample_times

            # Perform the SVD decomposition
            UA, sA, VA = np.linalg.svd(data_array_postprocess, full_matrices=True)
            VA = VA.T
            n, n = UA.shape

            cAmat = np.zeros((self.n_coeff, n))
            cAvar = np.zeros((self.n_coeff, n))
            for i in range(n):
                # Get the data matrix
                cAmat[:, i] = np.dot(
                    data_array_postprocess[i, :], VA[:, : self.n_coeff]
                )
                # Get variance matrix
                ErrorLevel = 1.0
                errors = ErrorLevel * np.ones_like(data_array_postprocess[i, :])
                cAvar[:, i] = np.diag(
                    np.dot(
                        VA[:, : self.n_coeff].T,
                        np.dot(np.diag(np.power(errors, 2.0)), VA[:, : self.n_coeff]),
                    )
                )
            cAstd = np.sqrt(cAvar)

            svd_model[filt]["n_coeff"] = self.n_coeff
            svd_model[filt]["cAmat"] = cAmat
            svd_model[filt]["cAstd"] = cAstd
            svd_model[filt]["VA"] = VA

        return svd_model

    def train_model(self):
        # Loop through filters
        for filt in self.filters:
            print("Computing Model for filter %s..." % filt)

            param_array_postprocess = self.svd_model[filt]["param_array_postprocess"]
            cAmat = self.svd_model[filt]["cAmat"]

            self.training_func(param_array_postprocess, cAmat, filt)

    def check_model(self):
        if not os.path.isfile(self.modelfile):
            return False
        try:
            for filt in self.filters:
                outfile = os.path.join(self.outdir, f"{filt}.{self.file_ending}")
                if not os.path.isfile(outfile):
                    return False
            ## we do not do this for api_gp-model and will fail as it has no file_ending
        except AttributeError:
            pass

        return True

    def save_model(self):
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        for filt in self.filters:
            outfile = os.path.join(self.outdir, f"{filt}.{self.file_ending}")
            self.save_routine(filt, outfile)

        joblib.dump(self.svd_model, self.modelfile, compress=9)

    def load_model(self):
        get_model(self.svd_path, f"{self.model}{self.model_specifier}", self.filters)
        self.svd_model = joblib.load(self.modelfile)

        for filt in self.svd_model.keys():
            self.load_routine(filt)
    
    def load_routine(self, filt):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_routine(self, filt, outfile):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def training_func(self, param_array_postprocess, cAmat, filt):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    
class KerasTrainingModel(BaseTrainingModel):
    def __init__(self, *args, **kwargs):

        self.model_specifier = ""
        self.file_ending = 'keras'
        super().__init__( *args, **kwargs)
        if self.plot and not os.path.isdir(self.plotdir):
            os.mkdir(self.plotdir)

    def load_routine(self, filt):
        outfile = os.path.join(self.outdir, f"{filt}.{self.file_ending}")
        self.svd_model[filt]["model"] = k.saving.load_model(outfile, compile=False)
        self.svd_model[filt]["model"].compile(optimizer="adam", loss="mse")

    def save_routine(self, filt, outfile):
        self.svd_model[filt]["model"].save(outfile)
        del self.svd_model[filt]["model"]

    def training_func(self, param_array_postprocess, cAmat, filt, dropout_rate=0.6):
        """
        Train a tensorflow model to emulate the KN model.
        """
        train_X, val_X, train_y, val_y = train_test_split(
            param_array_postprocess,
            cAmat.T,
            shuffle=True,
            test_size=0.1,
            random_state=self.random_seed,
        )

        k.utils.set_random_seed(self.random_seed)

        if self.model_exists and self.continue_training:
            model = self.svd_model[filt]["model"]
        else:
            model = k.Sequential()
            model.add(k.Input((train_X.shape[1],)))
            # One/few layers of wide NN approximate GP
            model.add(
                k.layers.Dense(
                    2048,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            model.add(k.layers.Dropout(dropout_rate))
            model.add(k.layers.Dense(self.n_coeff))

            # compile the model
            model.compile(optimizer="adam", loss="mse")

        # fit the model
        training_history = model.fit(
            train_X,
            train_y,
            epochs=self.n_epochs,
            batch_size=32,
            validation_data=(val_X, val_y),
            verbose=True,
        )

        if self.plot:
            loss = training_history.history["loss"]
            val_loss = training_history.history["val_loss"]
            plt.figure()
            plt.plot(loss, label="training loss")
            plt.plot(val_loss, label="validation loss")
            plt.legend()
            plt.xlabel("epoch number")
            plt.ylabel("number of losses")
            plt.savefig(
                os.path.join(self.plotdir, f"train_history_loss_{filt}.pdf")
            )
            plt.close()

        # evaluate the model
        error = model.evaluate(param_array_postprocess, cAmat.T, verbose=0)
        print(f"{filt} MSE:", error)

        self.svd_model[filt]["model"] = model

class TensorflowTrainingModel(KerasTrainingModel):
    """legacy class for compatibility with older tensorflow.keras-calls"""
    def __init__(self, *args,  **kwargs):
        super().__init__( *args, **kwargs)

        self.model_specifier = "_tf"
        self.file_ending = 'h5'

    
    def save_routine(self, filt, outfile):
        self.svd_model[filt]["model"].save(outfile, save_format=self.file_ending)
        del self.svd_model[filt]["model"]

class SklearnGPTrainingModel(BaseTrainingModel):
    def __init__(self, *args,  **kwargs):
        
        self.model_specifier = ""
        self.file_ending = 'joblib'
        super().__init__( *args, **kwargs)

    def load_routine(self, filt):
        outfile = os.path.join(self.outdir, f"{filt}.{self.file_ending}")
        if not os.path.isfile(outfile):
            return
        self.svd_model[filt]["gps"] = joblib.load(outfile)
    
    def save_routine(self, filt, outfile):
        joblib.dump(self.svd_model[filt]["gps"], outfile, compress=9)
        del self.svd_model[filt]["gps"]

    def training_func(self, param_array_postprocess, cAmat, filt):
        # Set of Gaussian Process
        kernel = 1.0 * RationalQuadratic(
            length_scale=1.0,
            alpha=0.1,
            length_scale_bounds=(1e-10, 1e10),
            alpha_bounds=(1e-10, 1e10),
        )

        print("Calculating the coefficents")

        def gp_func(cAmat_i):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
            gp.fit(param_array_postprocess, cAmat_i)
            return gp

        if self.ncpus > 1:
            from p_tqdm import p_map

            gps = p_map(gp_func, cAmat[: self.n_coeff, :], num_cpus=self.ncpus)
        else:
            gps = []
            for i in range(self.n_coeff):
                print("Coefficient %d/%d..." % (i + 1, self.n_coeff))
                gps.append(gp_func(cAmat[i, :]))

        self.svd_model[filt]["gps"] = gps

class GPAPITrainingModel(BaseTrainingModel):
    def __init__(self,  *args, **kwargs):
        self.model_specifier = "_api"
        super().__init__( *args, **kwargs)

    def load_routine(self, filt):
        for i, sub_model in enumerate(self.svd_model[filt]["gps"]):
            self.svd_model[filt]["gps"][i] = load_api_gp_model(sub_model)
    
    def save_model(self):
        get_model(self.svd_path, f"{self.model}_api", self.svd_model.keys())
        joblib.dump(self.svd_model, self.modelfile, compress=9)

    def training_func(self, param_array_postprocess, cAmat, filt):
        nd = 1
        # Construct hyperparamters
        coeffs = [0.5] * nd

        # Create the compact kernel
        kernel = CompactKernel.fit(param_array_postprocess, method="simple", 
                                 coeffs=coeffs, sparse=True)
        gps = []
        for i in range(self.n_coeff):
            # Fit the training data
            gp = GaussianProcess.fit(
                param_array_postprocess, cAmat[i, :], kernel=kernel, train_err=None
            )

            store = gp.store_options

            gp_dict = {}
            if gp.param_names is not None:
                gp_dict["param_names"] = ",".join(gp.param_names)
            else:
                gp_dict["param_names"] = None

            # Save the kernel
            gp_dict["kernel"] = gp.kernel.to_json()

            # Save basic attributes
            gp_dict["sparse"] = gp.sparse
            gp_dict["hypercube_rescale"] = gp.hypercube_rescale

            # Save any extra metadata
            gp_dict["metadata"] = json.dumps(gp.metadata)

            # Save the training error
            gp_dict["train_err"] = gp.train_err

            for option in store:
                if option == "x":
                    gp_dict["x"] = gp.x
                    gp_dict["y"] = gp.y
                elif option == "predictor":
                    gp_dict["predictor"] = gp.predictor
                else:
                    raise ValueError("Option should just be x or predictor")

            gps.append(gp_dict)

        self.svd_model[filt]["gps"] = gps



class SVDTrainingModel(object):
    def __init__(
        self,
        model,
        data,
        parameters,
        sample_times,
        filters,
        svd_path=None,
        n_coeff=10,
        n_epochs=15,
        interpolation_type="keras",
        data_type="photometry",
        data_time_unit="days",
        plot=False,
        plotdir=os.path.join(os.getcwd(), "plot"),
        ncpus=1,
        univariate_spline=False,
        univariate_spline_s=2,
        random_seed=42,
        start_training=True,
        continue_training=False,
    ):
        """interpolation_type: str, optional
        Type of interpolation, must be one keras (or explicitly its backends),sklearn_gp, api_gp"""
        # NOTE: This class is implemented for backwards compatibility. 
        # Directly initiating a KerasTrainingModel, SklearnGPTrainingModel, 
        # GPAPITrainingModel should be preferred.
        # Most of the previous attributes and methods are now implemented in BaseTrainingModel which again is subclassed subject to the interpolation_type. 
        # The __init__ creates a corresponding instance as a backend-attribute and
        # __getattr__ retrieves any properties or methods from them

        ##collect all arguments of __init__ to pass on
        setup_kwargs = locals()
        setup_kwargs.pop("self")
        ## set interpolation_type here, pass everything else
        self.interpolation_type = setup_kwargs.pop("interpolation_type")
        keras_backends = ["keras", "tensorflow", "jax", "torch"]
        if (interpolation_type not in keras_backends) and continue_training:
            print(
                "--continue-training only supported with --interpolation-type \
                 keras/tensorflow/jax/torch, this will have no effect"
            )
        if self.interpolation_type in keras_backends:
            try:
                ## We prefer keras over tensorflow, but can try the older fashion
                self.backend = KerasTrainingModel(**setup_kwargs)
            except:
                self.backend = TensorflowTrainingModel(**setup_kwargs)
        elif self.interpolation_type == "sklearn_gp":
            self.backend = SklearnGPTrainingModel(**setup_kwargs)
        elif self.interpolation_type == "api_gp":
            self.backend = GPAPITrainingModel(**setup_kwargs)
        else:
            raise ValueError(
                "interpolation_type must be sklearn_gp, api_gp or tensorflow"
            )

    # called when an attribute is not found, so almost always:
    def __getattr__(self, name):
        # We assume it is implemented in the backend
        return self.backend.__getattribute__(name)

        


def min_max_scaling(data):
    """
    row_wise Min-max scaling of data to [0, 1] range, assuming a 2d array as input
    """
    data = np.array(data)
    param_mins, param_maxs = np.min(data, axis=0), np.max(data, axis=0)
    rescaled_data = (data - param_mins) / (param_maxs - param_mins)
    return (rescaled_data, param_mins, param_maxs)  

def load_api_gp_model(gp):

    """Load a gaussian-process-api GaussianProcess model
    Parameters
    ----------
    gp : dict
        Dictionary representation of gaussian-process-api GaussianProcess model
    Returns
    -------
    gp_api.gaussian_process.GaussianProcess
    """

    from gp_api.gaussian_process import GaussianProcess
    from gp_api.kernels import from_json as load_kernel

    param_names = gp.get("param_names", None)
    if param_names is not None:
        param_names = param_names.split(",")

    sparse = gp["sparse"]
    hypercube_rescale = gp["hypercube_rescale"]

    metadata = json.loads(gp["metadata"])

    # TODO: actually figure out what can be loaded and what has to
    # be re-computed

    x = gp["x"]
    y = gp["y"]

    train_err = gp["train_err"]

    kernel = load_kernel(gp["kernel"])
    if train_err is not None:
        Kaa = kernel(x, x, train_err)
    else:
        Kaa = kernel(x, x)
    LL = GaussianProcess._get_cholesky(Kaa, sparse=sparse)

    predictor = gp["predictor"]

    return GaussianProcess(
        x,
        y,
        LL,
        predictor,
        kernel,
        hypercube_rescale=hypercube_rescale,
        param_names=param_names,
        metadata=metadata,
    )
