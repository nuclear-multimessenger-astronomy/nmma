import numpy as np
import shutil
from pathlib import Path
import json
import joblib
from ast import literal_eval
import keras as k
from ..core.conversion import (
    radii_from_qur,
    EOS_to_ns_parameters,
    EOS_to_system_parameters,
)


def setup_eos_generator(args):
    """Build the right EoSGenerator subclass for ``micro_eos_model``
    ("nep", "nep-5", "lec", "lec-7", or "lec-13"), reading its metadata
    from ``emulator_metadata`` -- a dict, a path to a JSON file, or a
    Python-literal dict string.

    Parameters
    ----------
    args: argparse.Namespace | dict
        Must provide ``micro_eos_model`` and ``emulator_metadata``
        (as attributes, or as dict keys if ``args`` is a plain dict).

    Returns
    -------
    EoSGenerator
    """
    if isinstance(args, dict):
        meta_dict = args
        eos_model_type = meta_dict["micro_eos_model"].lower()
    else:
        try:
            with open(args.emulator_metadata, "r") as f:
                meta_dict = json.load(f)
        except TypeError:
            meta_dict = args.emulator_metadata
        except FileNotFoundError:
            meta_dict = literal_eval(args.emulator_metadata)

        eos_model_type = args.micro_eos_model.lower()

    if eos_model_type == "nep":
        return NEPEoSGenerator(meta_dict)
    elif eos_model_type == "nep-5":
        return NEP5EoSGenerator(meta_dict)

    elif eos_model_type == "lec":
        return LECEoSGenerator(meta_dict)
    elif eos_model_type == "lec-7":
        return LEC7EoSGenerator(meta_dict)
    elif eos_model_type == "lec-13":
        return LEC13EoSGenerator(meta_dict)
    ## add more models
    else:
        raise ValueError(f"Unknown eos model type: {eos_model_type}")


class EoSGenerator:
    """Base on-the-fly EOS emulator: wraps a trained model (Keras, or a
    pickled sklearn-style model as fallback) that maps micro-EOS
    parameters to macroscopic (radius, mass, lambda) curves.

    Parameters
    ----------
    emulator_path: str
        Path to a Keras model, or a pickle file as fallback.
    eos_parameters: list of str, optional
        Names of the parameters to pass to the emulator, in order.
        Falls back to the class attribute of the same name if omitted.
    n_mass_samples: int, default 30
        Number of mass grid points per EOS; see ``set_mass_construction``.
    """
    eos_parameters = None

    def __init__(self, emulator_path, eos_parameters=None, n_mass_samples=30):

        # load the emulator
        try:
            self.emulator = k.saving.load_model(
                emulator_path, custom_objects=None, compile=False
            )

            # NOTE: in the main samling loop, the tensorflow-predict method is much slower
            #  than pure __call__ as it tries to cater to larger batches!
            # We therefore need to set the predict method in
            # accordance with the appropriate backend
            if k.backend.backend() == "tensorflow":
                self.predict = self.tensorflow_predict
            elif k.backend.backend() == "jax":
                self.predict = self.jax_predict
        except:
            import pickle

            with open(emulator_path, "rb") as f:
                self.emulator = pickle.load(f)
            self.predict = self.pickle_predict

        ## set the parameter-keys to be passed to the emulator
        if eos_parameters:
            self.eos_parameters = eos_parameters

        self.set_mass_construction(n_mass_samples)

    def pickle_predict(self, x):
        """``self.emulator.predict(x)``, for a plain pickled model."""
        return self.emulator.predict(x)

    def jax_predict(self, x):
        """``self.emulator.predict(x, verbose=0)``, for the jax Keras backend."""
        return self.emulator.predict(x, verbose=0)

    def tensorflow_predict(self, x):
        """``self.emulator(x)`` -- faster than ``.predict`` for the small
        batches used during sampling, on the tensorflow Keras backend."""
        return self.emulator(x)

    def set_mass_construction(self, n_mass_samples):
        """Use ``n_mass_samples`` equally-spaced mass grid points (1 to
        each EOS's TOV mass) per EOS, via ``equal_distance_masses``."""
        self.n_mass_samples = n_mass_samples
        self.decompose_mass_data = self.equal_distance_masses

    def equal_distance_masses(self, mtov):
        "Get mass array(s) from 1 to mtov of length n_mass_samples"
        mass_range = np.linspace(1, mtov, self.n_mass_samples, axis=-1)
        try:
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range

    def generate_macro_eos(self, converted_parameters):
        """Emulate and reshape one batch of EOS into (radius, mass,
        lambda) curves: ``emulate_macro_eos`` then ``adjust_format``."""
        predictions = self.emulate_macro_eos(converted_parameters)
        return self.adjust_format(predictions)

    def emulate_macro_eos(self, converted_parameters):
        """Run the emulator on ``converted_parameters``, via
        ``assemble_eos_params`` then ``self.predict``."""
        eos_params = self.assemble_eos_params(converted_parameters)
        return self.predict(eos_params)

    def assemble_eos_params(self, converted_parameters):
        """Assemble the parameters for the EoS model into a 2D-array for the emulator"""
        eos_params = np.array(
            [converted_parameters[par] for par in self.eos_parameters]
        ).T
        return np.atleast_2d(eos_params)

    def adjust_format(self, predictions):
        """Adjust the format of the predictions to the expected format: A n-tuple of three 1-D arrays for radius, mass, lambdas, respectively"""
        # This should be implemented in the subclass
        return predictions


class NEPEoSGenerator(EoSGenerator):
    """Nuclear-empirical-parameter EOS emulator.

    Parameters
    ----------
    metadata: dict
        Must include "emulator_path"; may include "backend" (asserted
        against the active Keras backend), "eos_parameters", and
        "n_mass_samples" (default 40) -- see ``set_mass_construction``.
    """
    def __init__(self, metadata):

        emulator_path = metadata["emulator_path"]
        if metadata.get("backend", False):
            assert (
                k.backend.backend() == metadata["backend"]
            ), f"Keras Backend mismatch: {k.backend.backend()} vs {metadata['backend']}. please set the environment variable KERAS_BACKEND to {metadata['backend']}"
        super().__init__(emulator_path, metadata.get("eos_parameters", None))

        n_mass_samples = metadata.get("n_mass_samples", 40)
        self.set_mass_construction(n_mass_samples)

    def set_mass_construction(self, n_mass_samples):
        """Use ``n_mass_samples`` equally-spaced mass points if it's an
        int (via ``equal_distance_masses``), or a disjoint low/high mass
        grid if it's a 2- or 3-tuple/list of
        ``(mass_samples_low, mass_samples_high[, split_value])`` (via
        ``disjoint_masses``; ``split_value`` defaults to 2.0)."""
        if isinstance(n_mass_samples, int):
            # if this is a single integer, use equally spaced masses
            self.n_mass_samples = n_mass_samples
            self.decompose_mass_data = self.equal_distance_masses
        elif isinstance(n_mass_samples, (tuple, list)):
            # iterable containing mass points for fixed-distance lower end, variably spaced upper end and optionally mass value at which these methods will be concatenated; if not given, the default is 2.0
            try:
                self.mass_samples_low, self.mass_samples_high, self.split_value = (
                    n_mass_samples
                )
            except ValueError:
                self.mass_samples_low, self.mass_samples_high = n_mass_samples
                self.split_value = 2.0
            self.n_mass_samples = self.mass_samples_low + self.mass_samples_high
            self.decompose_mass_data = self.disjoint_masses

    def disjoint_masses(self, mtov):
        """Helper function when using split mass construction.
        Some predicted TOV-masses may be lower than the split value
        for concatenation and would lead to unexpected behaviour.
        In that case we fall back to equally spaced mass arrays.
        However, this should not happen for EoSs with a physically
        reasonable TOV-mass and should only be seen as a graceful fallback.
        """
        return np.where(
            mtov > self.split_value,
            self.properly_disjoint_masses(mtov),
            self.equal_distance_masses(mtov),
        )

    def equal_distance_masses(self, mtov):
        "Get mass array(s) from 1 to mtov of length n_mass_samples"
        mass_range = np.linspace(1, mtov, self.n_mass_samples, axis=-1)
        try:
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range

    def properly_disjoint_masses(self, mtov):
        """Build a two-part mass grid: ``mass_samples_low`` points from 1
        to ``split_value``, plus ``mass_samples_high`` points from
        ``split_value`` (exclusive) up to ``mtov``. Used by
        ``disjoint_masses`` when ``mtov > split_value``."""
        mass_range_low = np.linspace(
            1, self.split_value * np.ones_like(mtov), self.mass_samples_low, axis=-1
        )
        mass_range_high = np.linspace(
            mtov, self.split_value, self.mass_samples_high, endpoint=False, axis=-1
        )
        mass_range_high = mass_range_high[..., ::-1]
        mass_range = np.concatenate([mass_range_low, mass_range_high], axis=-1)
        try:
            mass_range = np.squeeze(mass_range, axis=1)
        except ValueError:
            pass
        return mass_range

    def adjust_format(self, predictions):
        """Split the emulator's flat (radius | log10-lambda | TOV_mass)
        output into (radius, mass, lambda) curves, building the mass grid
        from the predicted TOV mass via ``self.decompose_mass_data``."""
        rad_data, lam_data, mtov_data = np.split(
            predictions, [self.n_mass_samples, 2 * self.n_mass_samples], axis=-1
        )
        mass_range = self.decompose_mass_data(mtov_data)

        lam_data = 10**lam_data

        return np.stack([rad_data, mass_range, lam_data], axis=1)


class NEP5EoSGenerator(NEPEoSGenerator):
    """NEPEoSGenerator using the 5 NEP coefficients K_sat, L_sym, K_sym,
    3n_sat, 5n_sat as emulator inputs (the ``--micro-eos-model nep-5``
    default)."""
    eos_parameters = ["K_sat", "L_sym", "K_sym", "3n_sat", "5n_sat"]


class LECEoSGenerator(EoSGenerator):
    """Chiral-EFT low-energy-constant EOS emulator: separate sklearn-style
    models (and scalers) for mass, radius, and lambda, rather than one
    combined Keras model. Does not call EoSGenerator.__init__.

    Parameters
    ----------
    metadata: dict
        Paths to joblib-pickled "feature_scaler", "lambda_scaler",
        "radius_scaler", "mass_emulator", "radius_emulator",
        "lambda_emulator", and optionally "n_mass_samples" (default 30).
    """
    def __init__(self, metadata):
        self.feature_scaler = joblib.load(metadata["feature_scaler"])
        self.lambda_scaler = joblib.load(metadata["lambda_scaler"])
        self.radius_scaler = joblib.load(metadata["radius_scaler"])

        self.mass_emulator = joblib.load(metadata["mass_emulator"])
        self.radius_emulator = joblib.load(metadata["radius_emulator"])
        self.lambda_emulator = joblib.load(metadata["lambda_emulator"])

        self.set_mass_construction(metadata.get("n_mass_samples", 30))

    def predict(self, converted_parameters):
        """Scale ``converted_parameters`` (the assembled 2D parameter
        array from ``assemble_eos_params``) and run the mass, radius,
        and lambda emulators on it.

        Returns
        -------
        tuple
            (mass_prediction, radius_prediction, lambda_prediction).
        """
        # Scale the input features
        scaled_features = self.feature_scaler.transform(converted_parameters)

        # Make predictions using the emulators
        mass_prediction = self.mass_emulator.predict(scaled_features)
        radius_prediction = self.radius_emulator.predict(scaled_features)
        lambda_prediction = self.lambda_emulator.predict(scaled_features)

        return (mass_prediction, radius_prediction, lambda_prediction)

    # def assemble_eos_params(self, converted_parameters):
    #     """Assemble the parameters for the EoS model into a 2D-array for the emulator"""
    #     eos_params = np.array([
    #         converted_parameters[par] + converted_parameters.get(f"{par}_shift", 0)
    #         for par in self.eos_parameters
    #     ]).T
    #     return np.atleast_2d(eos_params)

    def adjust_format(self, predictions):
        """Inverse-transform (mass, radius, lambda) via their scalers,
        building the mass grid from the predicted TOV mass via
        ``self.decompose_mass_data`` and exponentiating lambda."""
        mass_data, rad_data, lam_data = predictions

        # Inverse transform the predictions
        mass_array = self.decompose_mass_data(mass_data)
        radius_array = self.radius_scaler.inverse_transform(rad_data)
        lambda_array = self.lambda_scaler.inverse_transform(lam_data)
        return np.stack([radius_array, mass_array, 10**lambda_array], axis=1)


class LEC7EoSGenerator(LECEoSGenerator):
    """LECEoSGenerator using the 6 chiral-EFT low-energy constants
    d11, d22, d3, d4, d6, d7 as emulator inputs."""
    eos_parameters = ["d11", "d22", "d3", "d4", "d6", "d7"]


class LEC13EoSGenerator(LECEoSGenerator):
    """LECEoSGenerator using the 6 LEC7EoSGenerator coefficients plus 7
    saturation/speed-of-sound parameters (ksat, qsat, zsat, cssq1-4)."""
    eos_parameters = [
        "d11",
        "d22",
        "d3",
        "d4",
        "d6",
        "d7",
        "ksat",
        "qsat",
        "zsat",
        "cssq1",
        "cssq2",
        "cssq3",
        "cssq4",
    ]


class EoSConverter:
    """Parameter-conversion object that turns sampled EOS/mass parameters
    into macroscopic mass-radius-tidal-deformability curves and, for GW
    analyses, per-component lambda/radius.

    Auto-detects ``method`` from ``args`` if not given: "tabulated" if
    ``eos_file``/``eos_data`` is set, else "emulated" if
    ``emulator_metadata`` is set.

    Parameters
    ----------
    args: argparse.Namespace
        See ``eos_parsing.py`` for the relevant CLI args per method.
    method: str, optional
        "emulated" (on-the-fly generation via ``setup_eos_generator``),
        "tabulated" (a single ``eos_file``, or a directory/glob
        ``eos_data`` of files -- preloaded to RAM if ``eos_to_ram``, else
        loaded from disk on demand, renaming files to a canonical
        ``{1..N}.dat`` scheme first (see FIX ME comments below: this
        renaming step can currently crash or silently misassign files if
        ``eos_data`` isn't already named that way), or "qur" (skip EOS
        entirely, use quasi-universal relations via ``radii_from_qur``).
    """
    def __init__(self, args, method=None):
        if method is None:
            if getattr(args, "eos_file", None) or getattr(args, "eos_data", None):
                method = "tabulated"
            elif getattr(args, "emulator_metadata", None):
                method = "emulated"

        self.parameter_conversion = self.full_eos_conversion
        # Case 1: eos is generated from emulator on the fly
        if method == "emulated":
            self.tov_emulator = setup_eos_generator(args)
            self.macro_conversion = self.tov_emulator.generate_macro_eos

        elif method == "tabulated":
            # case 2: we use a single eos
            if getattr(args, "eos_file", None):
                self.eos_data = [np.loadtxt(args.eos_file, usecols=[0, 1, 2]).T]
                self.macro_conversion = self.single_eos_from_ram
                return

            # case 3 : we use multiple eos
            eos_path = Path(args.eos_data)
            if eos_path.is_dir():
                if getattr(args, "Neos", None) is None:
                    # FIX ME: Path.iterdir()'s order is arbitrary/OS-dependent,
                    # not sorted. Two distinct consequences from the same
                    # root cause, confirmed directly (os.listdir returned
                    # ['2.dat','3.dat','1.dat'] for 3 correctly-named files):
                    #  - Case 3b (eos_to_ram=False, below): combined with
                    #    the rename loop, this silently scrambles file
                    #    contents (each copied into the wrong {i+1}.dat
                    #    slot) -- confirmed all 3 files ended up holding
                    #    shuffled data.
                    #  - Case 3a (eos_to_ram=True): self.eos_data is built
                    #    directly in this order, so "EOS index i" doesn't
                    #    reliably mean the file actually named "{i+1}.dat"
                    #    -- confirmed a 2-EOS batch came back reversed.
                    # Needs `sorted(...)` here.
                    eos_files = list(eos_path.iterdir())
                else:
                    eos_files = [eos_path / f"{j+1}.dat" for j in range(args.Neos)]
            else:
                # FIX ME: glob() order is also not guaranteed sorted --
                # same risk as Path.iterdir() above.
                eos_files = list(Path().glob(args.eos_data))
                if getattr(args, "Neos", None):
                    assert args.Neos == len(
                        eos_files
                    ), "Number of EOS files found does not match Neos"

            self.Neos = len(eos_files)
            # Case 3a: precomputed eos data is loaded to ram
            if args.eos_to_ram:
                self.eos_data = [np.loadtxt(f, usecols=[0, 1, 2]).T for f in eos_files]
                self.macro_conversion = self.eos_from_ram

            # Case 3b: eos are loaded directly from file
            else:
                eos_dir = eos_files[0].parent
                for i, f in enumerate(eos_files):
                    # FIX ME: Path.samefile requires BOTH paths to
                    # already exist -- but that's precisely false when a
                    # file actually needs renaming (the canonical target
                    # doesn't exist yet), so this raises FileNotFoundError
                    # instead of returning False. Crashes by default
                    # (eos_to_ram=False) whenever eos_data isn't already
                    # named 1.dat, 2.dat, ...
                    if not f.samefile(eos_dir / f"{i+1}.dat"):
                        shutil.copy(f, eos_dir / f"{i+1}.dat")
                self.eos_data = eos_dir
                self.macro_conversion = self.eos_direct_load

        # case 4: no eos conversion, just QURs
        elif method == "qur":
            self.parameter_conversion = radii_from_qur
        else:
            raise ValueError(f"Unknown EoS conversion method: {method}")

    def __call__(self, parameters):
        """Forward to ``self.parameter_conversion``."""
        return self.parameter_conversion(parameters)

    def eos_direct_load(self, converted_parameters):
        """Load the requested EOS index/indices from disk, as
        ``{self.eos_data}/{EOS+1}.dat``.

        Parameters
        ----------
        converted_parameters: dict
            Must contain "EOS" (int or array of ints).

        Returns
        -------
        list of np.ndarray
            One (radius, mass, lambda) array per requested EOS.
        """
        EOSID = np.atleast_1d(converted_parameters["EOS"]).astype(int)
        return [
            np.loadtxt(self.eos_data / f"{j+1}.dat", usecols=[0, 1, 2]).T for j in EOSID
        ]

    def eos_from_ram(self, converted_parameters):
        """Look up the requested EOS index/indices in ``self.eos_data``
        (preloaded list), same interface as ``eos_direct_load``."""
        EOSID = np.atleast_1d(converted_parameters["EOS"]).astype(int)
        return [self.eos_data[i] for i in EOSID]

    def single_eos_from_ram(self, _):
        """Return the one preloaded EOS in ``self.eos_data``, ignoring the
        input (there's only ever one to choose from)."""
        return self.eos_data

    def full_eos_conversion(self, parameters):
        """``compute_macro_parameters`` then ``system_props_from_eos`` --
        the default ``parameter_conversion`` for GW+EOS joint analyses."""
        parameters = self.compute_macro_parameters(parameters)
        return self.system_props_from_eos(parameters)

    def compute_macro_parameters(self, parameters):
        """Convert the chosen EOS(s) into TOV_mass/TOV_radius/R_14/R_16
        via ``EOS_to_ns_parameters``, and stash the raw (radii, masses,
        lambdas) curve(s) in ``self.macro_parameters`` for later use
        (e.g. by ``system_props_from_eos``).

        Parameters
        ----------
        parameters: dict
            Must contain whatever ``self.macro_conversion`` needs
            (e.g. "EOS").

        Returns
        -------
        dict
            ``parameters``, with TOV_mass/TOV_radius/R_14/R_16 added.
        """
        eos_macro_keys = ["TOV_mass", "TOV_radius", "R_14", "R_16"]
        eos_data = self.macro_conversion(parameters)

        if len(eos_data) == 1:
            radii, masses, lambdas = eos_data[0]
            for key, val in zip(
                eos_macro_keys, EOS_to_ns_parameters(radii, masses, lambdas)
            ):
                parameters[key] = val
        else:
            radii, masses, lambdas = map(list, zip(*eos_data))
            TOV_mass_list, TOV_radius_list, R_14_list, R_16_list = [], [], [], []
            for rad, mass, lam in zip(radii, masses, lambdas):
                TOV_mass, TOV_radius, R_14, R_16 = EOS_to_ns_parameters(rad, mass, lam)
                TOV_mass_list.append(TOV_mass)
                TOV_radius_list.append(TOV_radius)
                R_14_list.append(R_14)
                R_16_list.append(R_16)
            for key, _list in zip(
                eos_macro_keys, [TOV_mass_list, TOV_radius_list, R_14_list, R_16_list]
            ):
                parameters[key] = np.array(_list)

        self.macro_parameters = {"radii": radii, "masses": masses, "lambdas": lambdas}
        return parameters

    def system_props_from_eos(self, converted_parameters):
        """Interpolate each component's tidal deformability and radius
        (lambda_1/2, radius_1/2) from ``self.macro_parameters``'s EOS
        curve(s), at its source-frame mass, via
        ``EOS_to_system_parameters``.

        Parameters
        ----------
        converted_parameters: dict
            Must contain "mass_1_source"/"mass_2_source".

        Returns
        -------
        dict
            ``converted_parameters``, with lambda_1/2, radius_1/2 added.
        """
        system_keys = ["lambda_1", "lambda_2", "radius_1", "radius_2"]

        m1_source = converted_parameters["mass_1_source"]
        m2_source = converted_parameters["mass_2_source"]
        radii, masses, lambdas = self.macro_parameters.values()

        if isinstance(radii, np.ndarray):  # single eos case
            for key, val_array in zip(
                system_keys,
                EOS_to_system_parameters(radii, masses, lambdas, m1_source, m2_source),
            ):
                converted_parameters[key] = val_array
        else:
            lambda_1_list, lambda_2_list, radius_1_list, radius_2_list = [], [], [], []
            for i, rad in enumerate(radii):
                lambda_1, lambda_2, radius_1, radius_2 = EOS_to_system_parameters(
                    rad, masses[i], lambdas[i], m1_source[i], m2_source[i]
                )

                lambda_1_list.append(lambda_1)
                lambda_2_list.append(lambda_2)
                radius_1_list.append(radius_1)
                radius_2_list.append(radius_2)

            for key, _list in zip(
                system_keys,
                [lambda_1_list, lambda_2_list, radius_1_list, radius_2_list],
            ):
                converted_parameters[key] = np.array(_list)

        return converted_parameters


def load_eos_files(eos_data, Neos):
    """Normalize ``eos_data`` into an explicit, sorted list of EOS files
    plus a resolved ``Neos`` count.

    Parameters
    ----------
    eos_data: str | list of str
        A directory path (globbed for "*.dat", sorted), or an
        already-resolved list of files.
    Neos: int | None
        If given, asserted to match the number of files found.

    Returns
    -------
    eos_data: list of str
    Neos: int
    """
    if isinstance(eos_data, str):
        eos_data = sorted(Path(eos_data).glob("*.dat"))
    if Neos:
        assert len(eos_data) == Neos, f"{eos_data} does not contain {Neos} eos-files"
    else:
        Neos = len(eos_data)
    return eos_data, Neos


def load_weights(weights):
    """Load ``weights`` from a file path via ``np.loadtxt``, or pass it
    through unchanged if it's not a string (already an array, or None)."""
    if isinstance(weights, str):
        return np.loadtxt(weights)
    else:
        return weights


def load_macro_characteristics_from_tabulated_eos_set(
    eos_data, Neos, masses_for_char_radii=None, masses_for_char_lambdas=None
):
    """utility to get MTOV and other characteristic properties for a set of tabulated EOS
    -----------
    Parameters:

    eos_data: str or list of strings
        if string: path to directory with eos_files, else list of eos_files to be read
    Neos: int
        Number of equations of state to consider
    masses_for_char_radii: int or tuple-like, default: None
        mass(es) at which characteristic radii should be evaluated
    masses_for_char_lambdas: int or tuple-like, default: None
        mass(es) at which characteristic tidal deformabilities should be evaluated

    --------
    Returns:
        output: list
            A list containing a 1d-array with the TOV-masses and optionally arrays
            with the characteristic radii and tidal deformabilities.

    FIX ME: currently broken -- crashes immediately (see below) even before
    reaching the missing `return output` at the end of the function, which
    would make it return None regardless. This backs the `combine-EOS`
    console script (nmma.post_processing.ns_characteristics.main), which
    is therefore also broken; no test currently covers it.
    """
    ####SETUP
    do_rads = False
    do_lams = False
    mtovs = np.empty(Neos)
    if masses_for_char_radii is not None:
        do_rads = True
        masses_for_char_radii = np.atleast_1d(masses_for_char_radii)
        # FIX ME: np.empty_like's 2nd positional arg is `dtype`, not `shape`
        # (that's np.empty's signature) -- this raises
        # "TypeError: Cannot interpret '<N>' as a data type" immediately.
        # Probably meant `np.empty((Neos, len(masses_for_char_radii)))`.
        radii = np.empty_like(Neos, len(masses_for_char_radii))
    if masses_for_char_lambdas is not None:
        do_lams = True
        masses_for_char_lambdas = np.atleast_1d(masses_for_char_lambdas)
        # FIX ME: same np.empty_like misuse as `radii` above.
        lambdas = np.empty_like(Neos, len(masses_for_char_lambdas))
    eos_data, Neos = load_eos_files(eos_data, Neos)

    ### Main Loop
    for i, eos_file in enumerate(eos_data):
        m, r, lam = np.loadtxt(eos_file, usecols=[1, 0, 2], unpack=True)
        mtovs[i] = m[-1]
        if do_rads:
            radii[i] = np.interp(masses_for_char_radii, m, r)
        if do_lams:
            lambdas[i] = np.interp(masses_for_char_lambdas, m, lam)

    output = [mtovs]
    if do_rads:
        output.append(np.squeeze(radii))
    if do_lams:
        output.append(np.squeeze(lambdas))
    # FIX ME: missing `return output` here -- the function falls off the
    # end and implicitly returns None, so the caller's tuple-unpacking
    # (e.g. `Mmax_prior, R14_prior = load_macro_characteristics_from...`)
    # would fail even if the np.empty_like crash above were fixed.

def load_tabulated_macro_eos_set_to_dict(eos_data, weights=None, Neos=None):
    """Load a tabulated EOS set into a dict keyed by EOS index (1-based),
    each holding its "R"/"M"/"Lambda" arrays and, if given, "weight".

    Parameters
    ----------
    eos_data: str | list of str
        See ``load_eos_files``.
    weights: str | array-like, optional
        See ``load_weights``.
    Neos: int, optional
        See ``load_eos_files``.

    Returns
    -------
    EOS_data: dict
    weights: np.ndarray or None
    Neos: int
    """
    eos_files, Neos = load_eos_files(eos_data, Neos)
    weights = load_weights(weights)

    EOS_data = {}
    for EOSIdx, eos_file in enumerate(eos_files):
        m, r, lam = np.loadtxt(eos_file, usecols=[1, 0, 2], unpack=True)
        EOS_data[EOSIdx + 1] = {"R": r, "M": m, "Lambda": lam}
        if weights is not None:
            EOS_data[EOSIdx + 1]["weight"] = weights[EOSIdx]

    return EOS_data, weights, Neos


# FIXME this should be used by conversion!
# CHECK ME: Can this be removed? It is not used anywhere.
def load_tabulated_macro_eos_set_to_list(eos_data, weights=None, Neos=None):
    eos_files, Neos = load_eos_files(eos_data, Neos)
    weights = load_weights(weights)

    EOS_data = [
        np.loadtxt(eos_file, usecols=[1, 0, 2], unpack=True) for eos_file in eos_files
    ]

    return EOS_data, weights, Neos
