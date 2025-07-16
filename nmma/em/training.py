import json
import os
import copy
import inspect
from glob import glob
import joblib
import warnings
import matplotlib.pyplot as plt
import numpy as np

from .utils import autocomplete_data, interpolate_nans, setup_sample_times
from ..utils.models import get_models_home, get_model  


from . import model_parameters
from .model import SVDLightCurveModel
from .io import read_training_data
from .em_parsing import parsing_and_logging, svd_training_parser, svd_model_benchmark_parser
from .plotting_utils import visualise_model_performance, chi2_hists_from_dict



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
        self.time_scale_factor = setup_time_conversion(data_time_unit)
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

        self.interpolate_data()

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


    def interpolate_data(self):
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
            obs_times = self.data[key]["t"]/ self.time_scale_factor

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

def SVDTrainingModel(
    *args,  
    interpolation_type="keras",
    **kwargs
    ):
    # NOTE: This function is implemented for backwards compatibility.
    # Directly initiating a KerasTrainingModel, SklearnGPTrainingModel,
    # GPAPITrainingModel should be preferred.
    
    
    keras_backends = ["keras", "tensorflow", "jax", "torch"]
    if interpolation_type in keras_backends:
        try:
            ## We prefer keras over tensorflow, but can try the older fashion
            return KerasTrainingModel(*args, **kwargs)
        except:
            return TensorflowTrainingModel(*args, **kwargs)
    elif interpolation_type == "sklearn_gp":
        return SklearnGPTrainingModel(*args, **kwargs)
    elif interpolation_type == "api_gp":
        return GPAPITrainingModel(*args, **kwargs)
    else:
        raise ValueError(
            "interpolation_type unknown, must be one of: keras, tensorflow, jax, torch, sklearn_gp, api_gp"
        )

      
def create_svdmodel():
    """Create a SVD model from command line arguments."""

    args = parsing_and_logging(svd_training_parser)
    svd_filenames = find_svd_files( args.data_path, args.ignore_bolometric )
    read_data = prepare_training_data(svd_filenames, args.format, args.data_type, args)
    training_data, parameters = create_svd_data( args.em_model,read_data)
    
    # filts = next(iter(training_data.values()))["lambda"] # for spectroscopy
    filts = setup_filters(args.filters, training_data, parameters)

    if args.axial_symmetry:
        training_data = axial_symmetry(training_data)

     # Specify the sample times in model training, implying a range of validity
    sample_times = setup_sample_times(args)

    training_args = [
        args.em_model,
        training_data,
        parameters,
        sample_times,
        filts,
        ]
    training_kwargs = dict(
            n_coeff=args.svd_mag_ncoeff,
            n_epochs=args.nepochs,
            svd_path=args.svd_path,
            data_type=args.data_type,
            data_time_unit=args.data_time_unit,
            plot=args.plot,
            plotdir=args.outdir,
            ncpus=args.ncpus,
            univariate_spline=args.use_UnivariateSpline,
            univariate_spline_s=args.UnivariateSpline_s,
            random_seed=args.random_seed,
            continue_training=args.continue_training,
    )
    try:
        training_model = KerasTrainingModel(*training_args, **training_kwargs)
    except:
        print("Your settings are not compatible with a keras training model.\n \
              Please consider adjusting your setup.\n \
            We will now try to train a legacy SVD model.")
        training_kwargs['interpolation_type'] = args.interpolation_type
        training_model = SVDTrainingModel(*training_args, **training_kwargs)

    #test-load the just trained model
    light_curve_model = SVDLightCurveModel(
        args.em_model,
        svd_path=args.svd_path,
        svd_mag_ncoeff=args.svd_mag_ncoeff,
        interpolation_type=args.interpolation_type,
        model_parameters=training_model.model_parameters,
        local_only=True,
    )
    if training_model.plot:
        visualise_model_performance(
            training_data, training_model, light_curve_model, args.data_type
        )
        
def benchmark():
    """Create a SVD model benchmark from command line arguments."""
    parser = svd_model_benchmark_parser()
    args = parser.parse_args()
    create_benchmark(**vars(args))
    
def create_benchmark(
    em_model,
    svd_path,
    data_path,
    format="bulla",
    interpolation_type="tensorflow",
    data_time_unit="days",
    svd_mag_ncoeff=10,
    tmin=None,
    tmax=None,
    filters=None,
    ncpus=1,
    outdir="benchmark_output",
    ignore_bolometric=True,
    local_only=False,
    plot= True
):
    """Create a benchmark for the SVD model.
    Parameters
    ----------
    em_model : str
        Name of the model to benchmark.
    svd_path : str
        Path to the svd directory.
    data_path : str
        Path to the data directory.
    format : str, optional
        Format of the data files. Default is "bulla".
    interpolation_type : str, optional
        Type of interpolation to use. Default is "tensorflow".
    data_time_unit : str, optional
        Unit of time for the data. Default is "days".
    svd_mag_ncoeff : int, optional
        Number of coefficients to use for the SVD model. Default is 10.
    tmin : float, optional
        Minimum time to consider for the benchmark. Default uses model_times.
    tmax : float, optional
        Maximum time to consider for the benchmark. Default uses model_times.
    filters : list, optional
        List of filters to use for the benchmark. If None, it will be determined from the data.
    ncpus : int, optional
        Number of CPUs to use for the benchmark. Default is 1.
    outdir : str, optional
        Output directory for the benchmark results. Default is "benchmark_output".
    ignore_bolometric : bool, optional
        Whether to ignore bolometric light curves in the data. Default is True.
    local_only : bool, optional
        Whether to use local files only. Default is False.
    plot : bool, optional
        Whether to plot the benchmark results. Default is True.
    """
    #### get the grid data file path
    # Implicitly set default ignore_bolometric as True for backward compatibility
    if ignore_bolometric is None:
        ignore_bolometric = True
    
    svd_filenames = find_svd_files(data_path, ignore_bolometric )
    read_data = prepare_training_data(svd_filenames, format)
    grid_training_data, parameters = create_svd_data(em_model,read_data)

    filts = setup_filters(filters, grid_training_data, parameters)
    time_scale_factor = setup_time_conversion(data_time_unit)
    # create the SVDlight curve model
    light_curve_model = SVDLightCurveModel(
        em_model,
        svd_path=svd_path,
        svd_mag_ncoeff=svd_mag_ncoeff,
        interpolation_type=interpolation_type,
        filters=filts,
        local_only=local_only,
    )
    if tmin is None:
        tmin = light_curve_model.model_times[0]
    if tmax is None:
        tmax = light_curve_model.model_times[-1]
    def chi2_func(grid_entry):
        grid_t = np.array(grid_entry["t"])/ time_scale_factor

        use_times = (grid_t > tmin) * (grid_t < tmax)

        # fetch the grid parameters
        parameter_entry = {param: grid_entry[param] for param in parameters}
        parameter_entry["redshift"] = 0.0

        # generate the corresponding light curve with SVD model
        estimate_mAB = light_curve_model.generate_lightcurve(
            grid_t[use_times], parameter_entry
        )
        # calculate chi2
        return {filt: np.nanmean(
                (np.array(grid_entry[filt])[use_times]  ##grid_mAB 
                - estimate_mAB[filt])**2
                )  for filt in filts}


    print(f"Benchmarking model {em_model} on filter {filts} with {ncpus} cpus")

    grid_entries = list(grid_training_data.values())
    if ncpus == 1:
        chi2_dict_array = [ chi2_func(entry) for entry in grid_entries ]
    else:
        from p_tqdm import p_map
        chi2_dict_array = p_map( chi2_func, grid_entries,  num_cpus=ncpus )

    chi2_array_by_filt = { filt: 
        [dict_entry[filt] for dict_entry in chi2_dict_array]
        for filt in filts } 

    # make the outdir
    model_subscript = "_tf" if interpolation_type == "tensorflow" else ""
    outpath = f"{outdir}/{em_model}{model_subscript}"
    os.makedirs(outpath, exist_ok=True)

    percentiles = [0, 25, 50, 75, 100]
    results_dct = { em_model: {
            filt:[np.round(np.percentile(chi2_array_by_filt[filt], val), 2) 
                            for val in percentiles] 
            for filt in filts
        } }
    outfile = f"{outpath}/benchmark_chi2_percentiles_{'_'.join(map(str, percentiles))}.json"
    with open(outfile, "w") as f:
        # save json file with filter-by-filter details
        json.dump(results_dct, f, indent=2)
    print(f"Saved file containing reduced chi2 percentiles at {outfile}.")
    print(f"Stats below are reduced chi2 distribution percentiles \
           {percentiles} for each filter:" )
    print(results_dct[em_model])

    if plot:
        chi2_hists_from_dict(chi2_array_by_filt, outpath)


def axial_symmetry(training_data):

    modelkeys = list(training_data.keys())
    if any(["KNtheta" not in training_data[key] for key in modelkeys]):
        raise ValueError("unknown symmetry parameter")

    for key in modelkeys:
        training = training_data[key]
        key_new = key + "_flipped"
        training_data[key_new] = copy.deepcopy(training)
        training_data[key_new]["KNtheta"] = -training_data[key_new]["KNtheta"]
        key_new = key + "_flipped_180"
        training_data[key_new] = copy.deepcopy(training)
        training_data[key_new]["KNtheta"] = 180 - training_data[key_new]["KNtheta"]

    return training_data

def find_svd_files(data_path, ignore_bolometric):
    """
    Set up the SVD data by finding all relevant files in the given data path.
    """

    link_string = '/*[!_Lbol].' if ignore_bolometric else '/*.'
    file_extensions = ["dat", "csv", "dat.gz", "h5"]
    
    filenames_lists = [glob(f"{data_path}{link_string}{ext}") for ext in file_extensions]
    filenames = [f for ext_list in filenames_lists for f in ext_list]
    if len(filenames) == 0:
        raise ValueError("Need at least one file to interpolate.")
    return filenames

def prepare_training_data( data_path, format="bulla", data_type="photometry", args = None):
    prelim_data = read_training_data(data_path, format, data_type, args)
    return interpolate_nans(prelim_data)

def create_svd_data(em_model, data):
    # create the SVD training data
    MODEL_FUNCTIONS = {
        k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
    }
    if em_model not in list(MODEL_FUNCTIONS.keys()):
        raise ValueError(
            f"{em_model} unknown. Please add to nmma.em.model_parameters"
        )
    model_function = MODEL_FUNCTIONS[em_model]
    return model_function(data)

def setup_filters(filters, training_data, parameters):

    # get the filts
    if isinstance(filters, str):
        filts = filters.replace(" ", "")  # remove all whitespace
        filts = filts.split(",")
    elif not filters:
        first_entry = next(iter(training_data.values()))
        filts = first_entry.keys() - set(["t"]+ parameters)
        filts = list(filts)
    else:
        # list input from analysis test code
        filts = filters

    if len(filts) == 0:
        raise ValueError("Need at least one valid filter.")
    return filts
    
def setup_time_conversion(data_time_unit="days"):
    """Set up the time conversion factor based on the data_time_unit."""
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
    return time_scale_factor

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
        x, y, LL,
        predictor, kernel,
        hypercube_rescale=hypercube_rescale,
        param_names=param_names, metadata=metadata,
    )
