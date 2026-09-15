from ast import literal_eval
from copy import copy
from pathlib import Path

import joblib
import numpy as np
import sncosmo
from astropy import units as u
from bilby.gw.cosmology import get_cosmology
from fiesta.inference.lightcurve_model import FluxModel
from scipy.special import logsumexp
from sncosmo.models import _SOURCES

from ..core.base import initialisation_args_from_signature_and_namespace
from ..core.constants import c_SI
from ..core.conversion import (
    distance_modulus_nmma,
    get_cosmo_grids,
    get_redshift,
    observation_angle_conversion,
)
from ..core.gitlab import get_model, get_models_home
from . import lightcurve_generation as lc_gen
from . import utils

ln10 = np.log(10)


# As different KN models have very different parameters,
# we need a dict to keep track for them. Remember, the order matters

model_parameters_dict = {
    # bolometric models
    "Arnett": ["tau_m", "log10_mni"],
    "Arnett_modified": ["tau_m", "log10_mni", "t_0"],
    # kilonova SVD models
    "Bu2019nsbh": ["log10_mej_dyn", "log10_mej_wind", "KNtheta"],
    "Bu2019lm": ["log10_mej_dyn", "log10_mej_wind", "KNphi", "KNtheta"],
    "Bu2019lm_sparse": ["log10_mej_dyn", "log10_mej_wind"],
    "Ka2017": ["log10_mej", "log10_vej", "log10_Xlan"],
    # GRB model
    "TrPi2018": [
        "inclination_EM",
        "log10_E0",
        "thetaCore",
        "thetaWing",
        "b",
        "L0",
        "q",
        "ts",
        "log10_n0",
        "p",
        "log10_epsilon_e",
        "log10_epsilon_B",
        "xi_N",
        "d_L",
    ],
    # Shock cooling models
    "Piro2021": ["log10_Menv", "log10_Renv", "log10_Ee"],
    # analytical kilonova models
    "Me2017": ["log10_mej", "log10_vej", "beta", "log10_kappa_r"],
    "HoNa2020": [
        "log10_mej",
        "vej_max",
        "vej_min",
        "vej_frac",
        "log10_kappa_low_vej",
        "log10_kappa_high_vej",
    ],
    "Bu2022mv": ["log10_mej_dyn", "vej_dyn", "log10_mej_wind", "vej_wind", "KNtheta"],
    # supernova models
    "PL_BB_fixedT": ["bb_luminosity", "temperature", "beta", "powerlaw_mag"],
    "blackbody_fixedT": ["bb_luminosity", "temperature"],
    "synchrotron_powerlaw": ["alpha_time", "beta_freq", "F_ref", "luminosity_distance"],
    "CV": ["example_num"],
    "AnBa2022_sparse": ["mrp", "xmix"],
    "AnBa2022_log": ["log10_mtot", "log10_mni", "vej", "log10_mrp", "xmix"],
    "AnBa2022_linear": ["mtot", "mni", "vej", "mrp", "xmix"],
    "salt2": ["x0", "x1", "c"],
    # for Sr2023, the following array is not used
    "Sr2023": ["a_AG", "alpha_AG", "f_nu_host"],
    # surrogate KN models
    "Bu2022Ye": [
        "log10_mej_dyn",
        "vej_dyn",
        "Yedyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "Bu2023Ye": [
        "log10_mej_dyn",
        "vej_dyn",
        "Yedyn",
        "log10_mej_wind",
        "vej_wind",
        "Yewind",
        "KNtheta",
    ],
    "LANL2022": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTP1": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTP2": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTS1": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
    "LANLTS2": [
        "log10_mej_dyn",
        "vej_dyn",
        "log10_mej_wind",
        "vej_wind",
        "KNtheta",
    ],
}

citation_dict = {
    # Supernova models
    **dict.fromkeys(
        ["AnBa2022_sparse", "AnBa2022_log", "AnBa2022_linear"],
        ["https://arxiv.org/abs/2302.09226", "https://arxiv.org/abs/2205.10421"],
    ),
    "nugent-hyper": ["https://arxiv.org/abs/astro-ph/0403450"],
    # SVD kilonova models
    **dict.fromkeys(
        ["Bu2019lm", "Bu2019lm_sparse"],
        ["https://arxiv.org/abs/2002.11355", "https://arxiv.org/abs/1906.04205"],
    ),
    "Bu2019nsbh": [
        "https://arxiv.org/abs/2009.07210",
        "https://arxiv.org/abs/1906.04205",
    ],
    **dict.fromkeys(
        ["Bu2022Ye", "Bu2023Ye", "Bu2022mv"],
        ["https://arxiv.org/abs/2307.11080", "https://arxiv.org/abs/1906.04205"],
    ),
    "Ka2017": ["https://arxiv.org/abs/1710.05463"],
    **dict.fromkeys(
        ["LANLTP1", "LANLTP2", "LANLTS1", "LANLTS2"],
        ["https://arxiv.org/abs/2105.11543"],
    ),
    # analytical kilonova models
    "Me2017": ["https://arxiv.org/abs/1910.01617"],
    "HoNa2020": ["https://arxiv.org/abs/1909.02581", "https://arxiv.org/abs/1206.2379"],
    **dict.fromkeys(
        ["PL_BB_fixedT", "blackbody_fixedT", "synchrotron_powerlaw"],
        ["Analytical models"],
    ),
    # Shock cooling model
    "Piro2021": ["https://arxiv.org/abs/2007.08543"],
    # Host Galaxy model
    "Sr2023": ["https://arxiv.org/pdf/2303.12849"],
    # GRB model
    "TrPi2018": ["https://arxiv.org/abs/1909.11691"],
}


class LightCurveModelContainer:
    """A parent-class for light curve model objects to evaluate lightcurves
    across filters with a set of parameters given

    Parameters
    ----------
    model: str
        Name of the model
    filters : List[str]
        List of filters to create model for.
        Defaults to all available filters.
    model_parameters: list, None
        list of alternative model parameters, if not specified default will be used.
    sample_times: array_like, optional
        Times at which to sample the light curve. If None, sets model default.
    extinction_model: str or None, optional
        Name of the extinction model to use. Defaults to "P92_SMC_host".
    **kwargs
        Additional keyword arguments accessible to the model.

    Returns
    -------
    LightCurveModel: `nmma.em.model.LightCurveModelContainer`
        A light curve model object to compute light curves
        from a set of parameters
    """

    def __init__(
        self,
        model,
        filters=None,
        model_parameters=None,
        sample_times=None,
        extinction_model=None,
        **kwargs,
    ):
        self.model = model
        self.identify_model_parameters(model_parameters)
        self.redshift_func = get_redshift
        if isinstance(filters, str):
            filters = filters.split(",")
        self.filters = filters
        self.default_filts, self.lambdas = utils.get_default_filts_lambdas(self.filters)
        self.nu_0s = c_SI / self.lambdas
        self.good_parameters = True
        self.extinction_model = extinction_model
        self.extinction_frame = None

        # sample times are used as nodes to generate the light curve,
        # characterising the model's validity range and the resolution
        # below which interpolation should not be performed.
        self.model_times = (
            sample_times if sample_times is not None else self.setup_model_times()
        )
        self.kwargs = kwargs

    def __repr__(self):
        return self.__class__.__name__ + f"(model={self.model})"

    def identify_model_parameters(self, model_parameters):
        """Identify the model parameters if they have not been provided.

        Parameters
        ----------
        parameters: None or dict
            Parameters to identify.

        """
        if model_parameters is None:
            assert self.model in model_parameters_dict.keys(), (
                f"{self.model} unknown,"
                "please update model_parameters_dict at em/model.py"
            )
            self.model_parameters = model_parameters_dict[self.model]
        else:
            self.model_parameters = model_parameters

    def setup_model_times(self, tmin=0.01, tmax=14.0, nsteps=150):
        """Build the time grid the model is evaluated on.

        These times define both the range over which the model is trusted
        and the resolution below which interpolating it makes no sense.
        Subclasses override this to follow whatever their own model covers.

        Parameters
        ----------
        tmin: float, optional
            First time, in days.
        tmax: float, optional
            Last time, in days.
        nsteps: int, optional
            Number of steps, spaced geometrically.

        Returns
        -------
        numpy.ndarray
            The time grid, in days.
        """
        return np.geomspace(tmin, tmax, nsteps)

    def check_vs_priors(self, priors):
        """Warn about missing priors, and prepare what they imply.

        Beyond warning, this sets up two things the priors decide: how a
        redshift will be obtained when only a luminosity distance is
        sampled, and which extinction model to apply when Ebv is.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.
        """
        for key in self.model_parameters:
            if key not in priors:
                print(f"Parameter {key} not found in priors, might fail.")

        if (
            "redshift" not in priors
            and "luminosity_distance" in priors
            and "Hubble_constant" not in priors
        ):
            dlum_prior = priors["luminosity_distance"]
            cosmo = getattr(dlum_prior, "cosmology", get_cosmology())
            dist_grid, z_grid = get_cosmo_grids(
                dlum_prior.minimum, dlum_prior.maximum, cosmo
            )

            def redshift_from_dlum(parameters):
                if "redshift" in parameters:
                    return parameters["redshift"]
                return np.interp(
                    parameters["luminosity_distance"], dist_grid, z_grid
                ).value

            self.redshift_func = redshift_from_dlum

        if "Ebv" in priors:
            ext_model, frame = utils.get_extinction_model(self.extinction_model)
            if self.extinction_frame is None:
                # for efficient computation of extinction corrections
                self.wavenumbers = 1.0 / (self.lambdas * u.meter.to(u.micron))
                if frame == "rest":
                    self.extinction_wavenumbers = self.rest_wavenumbers
                elif frame == "obs":
                    self.extinction_wavenumbers = self.obs_wavenumbers
                self.extinction_frame = frame
                self.extinction_model = ext_model
                self.extinction_range = self.extinction_model.x_range

    def sanity_checks(self, parameters):
        """Flag whether the model can be trusted for these parameters.

        Always passes here; subclasses that have unphysical corners
        override it.

        Parameters
        ----------
        parameters: dict
            Parameters about to be evaluated.
        """
        self.good_parameters = True

    def parameter_conversion(self, parameters):
        """Fill in the parameters the model wants from those sampled.

        A model may ask for a quantity in log space while the sampler works
        in linear space, or the other way round; both are derived here.
        Missing parameters are left alone, so that a later stage can still
        supply them.

        Parameters
        ----------
        parameters: dict
            Sampled parameters.

        Returns
        -------
        dict
            The parameters, with the model's own ones added.
        """

        new_parameters = observation_angle_conversion(parameters)
        for key in self.model_parameters:
            if key not in new_parameters:
                if key.lstrip("log10_") in new_parameters.keys():
                    new_parameters[key] = np.log10(new_parameters[key.lstrip("log10_")])
                elif "log10_" + key in new_parameters.keys():
                    new_parameters[key] = 10 ** new_parameters["log10_" + key]
                else:
                    pass  # Unclean fix, allows later addition of required params

        self.sanity_checks(new_parameters)
        return new_parameters

    def em_parameter_setup(self, parameters, combine_params=True):
        """Cache the quantities every filter evaluation will need.

        Distance, extinction, timeshift and redshift are the same for all
        filters and all times, and the redshift in particular is expensive,
        so they are computed once here and stored on the object.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.
        combine_params: bool, optional
            If True, also return the parameters the model itself expects.

        Returns
        -------
        dict or None
            The model parameters, when combine_params is set.
        """

        # read here, but used later for correction that is observation-dependent
        self.Ebv = parameters.get("Ebv", 0.0)
        self.luminosity_distance = parameters.get(
            "luminosity_distance", 1e-5
        )  # default 10pc = 1e-5 Mpc
        self.distmod = distance_modulus_nmma(
            self.luminosity_distance
        )  # default 10pc = 1e-5 Mpc
        self.timeshift = parameters.get("timeshift", 0.0)

        # redshift computation can be expensive, so we only do it
        # once and store it for conversion to detector frame
        self.redshift = self.redshift_func(parameters)
        if combine_params:
            return self.combine_lc_params(parameters)

    def combine_lc_params(self, parameters):
        """Extract the light curve parameters in the necessary format.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Combined light curve parameters.
        """
        return {
            key: parameters[key] if key in parameters else getattr(self, key)
            for key in self.model_parameters
        }

    def rest_wavenumbers(self):
        """Filter wavenumbers as seen in the rest frame of the source.

        Returns
        -------
        numpy.ndarray
            Wavenumbers in 1/micron.
        """

        return self.wavenumbers * (1 + self.redshift)

    def obs_wavenumbers(self):
        """Filter wavenumbers as seen by the observer.

        Returns
        -------
        numpy.ndarray
            Wavenumbers in 1/micron.
        """

        return self.wavenumbers

    def extinction_correction(self, model_mags):
        """Dim the model magnitudes by interstellar extinction.

        Filters whose wavenumber falls outside the range the extinction
        model covers are left untouched rather than extrapolated.

        Parameters
        ----------
        model_mags: dict
            Magnitude per filter, modified in place.

        Returns
        -------
        dict
            The same magnitudes, extinguished.
        """

        ext_factor = np.ones_like(self.lambdas)
        wavenumbers = self.extinction_wavenumbers()
        k_min, k_max = self.extinction_range
        covered = (wavenumbers >= k_min) & (wavenumbers <= k_max)
        ext_factor[covered] = self.extinction_model.extinguish(
            wavenumbers[covered], Ebv=self.Ebv
        )
        ext_mags = -2.5 * np.log10(ext_factor)
        for ext_mag, filt in zip(ext_mags, self.default_filts):
            if filt in model_mags:
                model_mags[filt] += ext_mag
        return model_mags

    def gen_detector_lc(self, parameters=None, sample_times=None):
        """Generate the light curve as it would be seen by a detector.

        The model works in the source frame. Its times are stretched by the
        redshift and offset by the timeshift, and its magnitudes are moved
        to the observer by the distance modulus and the extinction.

        Parameters
        ----------
        parameters: dict, optional
            Parameters of the light curve model.
        sample_times: numpy.ndarray, optional
            Source-frame times to evaluate. Defaults to the model's grid.

        Returns
        -------
        tuple
            Observer-frame times in days, and the apparent magnitude per
            filter.
        """
        if sample_times is None:
            sample_times = self.model_times

        model_lc = self.generate_lightcurve(sample_times, parameters)

        # redshift has been set in em_parameter_setup
        # timeshift is a detector-frame correction parameter
        observable_times = sample_times * (1.0 + self.redshift) + self.timeshift

        return self.combine_detector_data(model_lc, observable_times)

    def generate_lightcurve(self, sample_times, parameters):
        """Evaluate the model in the source frame. Subclasses must define it.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Absolute magnitude per filter.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def combine_detector_data(self, model_lc, observable_times):
        """Turn source-frame magnitudes into what a detector would record.

        Applies extinction, the distance modulus, and the correction for the
        fact that a redshifted source is integrated over a shifted
        luminosity window. Filters the model could not evaluate anywhere are
        returned as non-detections.

        Parameters
        ----------
        model_lc: dict
            Absolute magnitude per filter, in the source frame.
        observable_times: numpy.ndarray
            Observer-frame times, in days.

        Returns
        -------
        tuple
            The times, and the apparent magnitude per filter.
        """

        if self.extinction_frame:
            model_lc = self.extinction_correction(model_lc)

        # abs_mags consider source frame fluxes, so we have to correct
        # for the fact that we integrate over the 'wrong' luminosity window
        redshift_correction = -2.5 * np.log10(1.0 + self.redshift)

        lc_data = {}
        for filt, mags in model_lc.items():
            if np.isfinite(mags).any():
                apparent_magnitude = mags + self.distmod + redshift_correction
            else:  # no meaningful inter-/extrapolation possible
                apparent_magnitude = np.full_like(observable_times, np.inf)
            lc_data[filt] = apparent_magnitude

        return (observable_times, lc_data)

    @property
    def citation(self):
        """The reference to cite for this model.

        Returns
        -------
        dict
            The model name, mapped to its references.
        """
        return {self.model: citation_dict[self.model]}


class FiestaModel(LightCurveModelContainer):
    """Wrap a fiesta surrogate as an NMMA light curve model.

    fiesta returns light curves already in the observer frame, so this class
    short-circuits most of the frame handling the base class does.
    """

    load_dir_string = None

    def __init__(
        self, model, filters=None, surrogate_dir=None, sample_times=None, **kwargs
    ):
        """A light curve model object for evaluating light curves using fiesta.

        Parameters
        ----------
        model: str
            Name of the fiesta surrogate model to load.
        filters: str or list of str, optional
            Filters to use for the light curve. Defaults to all trained filters.
        surrogate_dir: str, optional
            Path to the directory containing the surrogate models.
        sample_times: array_like, optional
            Unused, included for compatibility with other Models.
        **kwargs
            Additional keyword arguments for LightCurveModelContainer, might be unused.
        """

        # fiesta requires a non-empty in-range filter list at construction;
        # there is no "all trained filters" default. Pick a safe optical/NIR
        # set that fits every published Bu* surrogate.
        # FIXME: investigate better ways to do this
        filters = (
            list(filters)
            if filters
            else [
                "sdssg",
                "sdssr",
                "sdssi",
                "sdssz",
                "ztfg",
                "ztfr",
                "ztfi",
                "2massj",
                "2massh",
                "2massks",
            ]
        )

        fiesta_kwargs = dict(
            name=model,
            filters=filters,
            directory=surrogate_dir,
        )
        try:
            self.fiesta_model = FluxModel(**fiesta_kwargs)
        except OSError:
            fiesta_kwargs["directory"] = Path(
                surrogate_dir, self.load_dir_string, model, "model"
            )
            self.fiesta_model = FluxModel(**fiesta_kwargs)
        if sample_times is not None:
            print("Warning: sample_times are not used in FiestaModel, ignoring.")
        kwargs["model_parameters"] = self.fiesta_model.parameter_names
        super().__init__(self.fiesta_model.name, filters, **kwargs)

    def setup_model_times(self):
        """Use the time grid the fiesta surrogate was trained on.

        Returns
        -------
        numpy.ndarray
            The surrogate's own times, in days.
        """
        return self.fiesta_model.times  # default sample times for fiesta model

    def check_vs_priors(self, priors):
        """Refuse priors that reach outside what the surrogate was trained on.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.

        Raises
        ------
        ValueError
            If a prior extends beyond the range of the training data.
        """
        super().check_vs_priors(priors)
        for key in self.model_parameters:
            try:
                prior = priors[key]
                param_info = self.fiesta_model.parameter_distributions[key]
                if (prior.minimum < param_info[0]) or (prior.maximum > param_info[1]):
                    raise ValueError(
                        f"Parameter {key} has bounds {param_info}, but prior was {prior}."
                    )

            except KeyError:
                print(f"Parameter {key} not found in priors, might fail.")

    def combine_lc_params(self, parameters):
        """Hand fiesta the frame parameters it expects to handle itself.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model, modified in place.

        Returns
        -------
        dict
            The same parameters, with redshift, distance and timeshift added.
        """
        parameters["redshift"] = self.redshift
        parameters["luminosity_distance"] = self.luminosity_distance
        parameters["timeshift"] = self.timeshift
        return parameters

    def gen_detector_lc(self, parameters=None, sample_times=None):
        """Generate the observer-frame light curve, straight from fiesta.

        Unlike the base class, no redshift stretching is applied: fiesta
        already returns observer-frame times. Only the timeshift is added.

        Parameters
        ----------
        parameters: dict, optional
            Parameters of the light curve model.
        sample_times: numpy.ndarray, optional
            Unused, kept for compatibility with the other models.

        Returns
        -------
        tuple
            Observer-frame times in days, and the magnitude per filter. The
            magnitudes are empty if the parameters were rejected.
        """
        # convert the parameters to the fiesta model parameters
        parameters = self.em_parameter_setup(parameters)
        if self.good_parameters:
            # generate the light curve using fiesta
            time_range, mag = self.fiesta_model.predict(parameters)

            if self.extinction_frame:
                mag = self.extinction_correction(mag)

            # we are in observer frame, but still need to add the timeshift
            # time_range = is in jax-specific format that we need to convert
            return (np.array(time_range) + self.timeshift, mag)
        else:
            return self.fiesta_model.times, {}

    def generate_lightcurve(self, sample_times, parameters):
        """Recover the source-frame light curve from the observer-frame one.

        fiesta works in the observer frame, so this undoes the redshift and
        the distance modulus to satisfy the base class interface. Prefer
        gen_detector_lc, which skips the round trip.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Absolute magnitude per filter.
        """

        obs_times, obs_mags = self.gen_detector_lc(parameters)

        # obs_times were redshift corrected; have to reverse this:
        source_times = (obs_times - self.timeshift) / (1 + self.redshift)
        abs_mags = {
            filt: np.interp(sample_times, source_times, obs_mag - self.distmod)
            for filt, obs_mag in obs_mags.items()
        }

        return abs_mags


class SimpleBolometricLightCurveModel(LightCurveModelContainer):
    """A light curve model object for Arnett's bolometric supernova light curves.

    Parameters
    ----------
    model: string, optional
        Name of the model. Can be either "Arnett" (default) or "Arnett_modified"
    **kwargs
        Additional keyword arguments accessible to the model.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SimpleBolometricLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters
    """

    def __init__(self, model="Arnett", **kwargs):
        super().__init__(model, **kwargs)
        if model == "Arnett":
            self.lc_func = lc_gen.arnett_lc
        elif model == "Arnett_modified":
            self.lc_func = lc_gen.arnett_modified_lc

    def setup_model_times(self):
        """Use a linear grid stopping where the integrals start to diverge.

        Returns
        -------
        numpy.ndarray
            Times from 0.005 to 20 days.
        """
        tmin = 0.005  # minimum time in days, arbitrary
        tmax = 20.0  # NOTE: The underlying integrals tend to diverge at later times
        nsteps = 40  # number of time steps

        return np.linspace(tmin, tmax, nsteps)

    def combine_detector_data(self, model_lc, observable_times):
        """Redshift a bolometric luminosity rather than a magnitude.

        A luminosity is energy per unit time, and both the energy and the
        time bin are stretched, hence the squared factor.

        Parameters
        ----------
        model_lc: numpy.ndarray
            Source-frame bolometric luminosity, in erg/s.
        observable_times: numpy.ndarray
            Observer-frame times, in days.

        Returns
        -------
        tuple
            The times, and the observed luminosity.
        """
        # lBol is essentially energy per time bin, both quantities
        #  get a redshift correction in detector frame
        return (observable_times, model_lc / (1 + self.redshift) ** 2)

    def generate_lightcurve(self, sample_times, parameters):
        """Evaluate the bolometric model.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        numpy.ndarray
            Bolometric luminosity, in erg/s.
        """
        new_parameters = self.em_parameter_setup(parameters)
        return self.lc_func(sample_times, new_parameters)


class SVDLightCurveModel(LightCurveModelContainer):
    """A light curve model object for evaluating light curves using prebuilt SVD models.

    An object to evaluate the light curve across filters
    with a set of parameters given based on a prebuilt SVD.

    Parameters
    ----------
    model: str
        Name of the model.
    svd_path: str, optional
        Path to the SVD directory.
    svd_mag_ncoeff: int, optional
        Number of highest eigenvalues to be taken for magnitude's SVD evaluation.
    svd_lbol_ncoeff: int, optional
        Number of highest eigenvalues to be taken for bolometric luminosity's SVD evaluation.
    interpolation_type: str, optional
        Type of interpolation to use. Can be 'keras','sklearn_gp', 'api_gp', or a keras-backend.
    model_parameters: list, optional
        List of alternative model parameters. If not specified, default will be used.
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    local_only: bool, optional
        If True, only local models will be used.
    **kwargs: optional
        Additional keyword arguments, including specifications for the underlying emulator.
    Returns
    -------
    LightCurveModel: `nmma.em.model.SVDLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    def __init__(
        self,
        model,
        svd_path=None,
        svd_mag_ncoeff=None,
        svd_lbol_ncoeff=None,
        interpolation_type="keras",
        filters=None,
        local_only=False,
        **kwargs,
    ):
        # Some models have underscores. Keep those, but drop '_tf' if it exists
        model_name_components = model.split("_")
        if "tf" in model_name_components:
            model_name_components.remove("tf")

        core_model_name = "_".join(model_name_components)

        self.mag_ncoeff = svd_mag_ncoeff
        self.lbol_ncoeff = svd_lbol_ncoeff
        self.interpolation_type = interpolation_type
        self.svd_path = get_models_home(svd_path)

        modelfile = self.svd_path / f"{core_model_name}.joblib"
        if interpolation_type == "tensorflow":
            self.model_specifier = "_tf"
        else:
            self.model_specifier = ""
        if not local_only:
            # FIXME Does this make sense for api_gp, too?
            filters = self.get_model_data(core_model_name, filters)

        super().__init__(core_model_name, filters, **kwargs)
        try:
            svd_mag_model = joblib.load(modelfile, **self.kwargs)
            # temporary fix, moving towards permament setting of sncosmo filter names
            self.svd_mag_model = {
                k.replace("_", ":"): v for k, v in svd_mag_model.items()
            }
            self.svd_lbol_model = None  # FIXME: this is not yet implemented

            # reset necessary after loading the model
            self.model_times = self.setup_model_times()

        except ValueError:
            raise ValueError(
                "Model file not found: {modelfile}\n \
                If possible, try removing the --local-only flag and rerun."
            )

        if self.filters is None:
            try:
                self.filters = list(self.svd_mag_model.keys())
            except TypeError:  # if using a bol_model, we keep None
                pass

        if self.interpolation_type == "sklearn_gp":
            self.load_filt_model(model, joblib.load, fn_ext="joblib", target_name="gps")

        elif self.interpolation_type == "api_gp":
            from .training import load_api_gp_model

            for filt in self.filters:
                for ii, gp_model in enumerate(self.svd_mag_model[filt]["gps"]):
                    self.svd_mag_model[filt]["gps"][ii] = load_api_gp_model(gp_model)

        elif self.interpolation_type in ("keras", "tensorflow", "torch", "jax"):
            import keras as k

            def keras_load_model(model_file):
                return k.saving.load_model(model_file, compile=False)

            try:
                self.load_filt_model(
                    self.model, keras_load_model, fn_ext="keras", target_name="model"
                )
            # if no filter- model is found, try to load the legacy h5 model
            except ValueError:
                self.load_filt_model(
                    self.model, keras_load_model, fn_ext="h5", target_name="model"
                )

        else:
            raise ValueError(
                "--interpolation-type must be sklearn_gp, api_gp or tensorflow"
            )

    def setup_model_times(self):
        """Use the time grid the surrogate was trained on, if it has one.

        Returns
        -------
        numpy.ndarray
            The surrogate's times, or the base class default.
        """
        # reset model_times to the model's training times if possible
        try:
            return next(iter(self.svd_mag_model.values()))["tt"]
        except Exception:
            return super().setup_model_times()

    def get_model_data(self, model, filters):
        """Fetch the surrogate from the model repository.

        Filter names are translated on the way out and back: the repository
        stores ``ps1_g`` where NMMA writes ``ps1::g``.

        Parameters
        ----------
        model: str
            Name of the model to download.
        filters: str or list of str or None
            Filters to fetch. All available ones when None.

        Returns
        -------
        list of str
            Filters the surrogate actually provides.
        """
        # we need to map the filter names to repo names!
        search_filters = None
        if isinstance(filters, str):
            filters = filters.split(",")
        if filters is not None:
            search_filters = [filt.replace(":", "_") for filt in filters]
        _, model_filters = get_model(
            self.svd_path, f"{model}{self.model_specifier}", filters=search_filters
        )
        if filters is None and model_filters is not None:
            filters = [filt.replace("_", ":") for filt in model_filters]
        return filters

    def load_filt_model(self, model, load_method, fn_ext="joblib", target_name="gps"):
        """Load the trained predictor of every filter into the surrogate.

        The model file alone carries only metadata. This attaches the actual
        estimator, one per filter, so the surrogate can be evaluated.

        Parameters
        ----------
        model: str
            Name of the model whose files to load.
        load_method: callable
            Loader matching the file type, joblib or keras.
        fn_ext: str, optional
            Extension of the files to look for.
        target_name: str, optional
            Key the estimator is stored under, per filter.

        Raises
        ------
        ValueError
            If no file was found for any filter at all. Missing individual
            filters only produce a warning.
        """
        outdir = self.svd_path / f"{model}{self.model_specifier}"
        found_any_model = False
        not_found = []
        for filt in self.filters:
            outfile = outdir / f"{filt.replace(':', '_')}.{fn_ext}"
            if outfile.exists():
                self.svd_mag_model[filt][target_name] = load_method(outfile)
                found_any_model = True
            else:
                not_found.append(filt)
        if not found_any_model:
            raise ValueError(f"No {fn_ext}-model files found for {model} in {outdir}")
        elif len(not_found) > 0:
            print(
                f"Warning: No {fn_ext}-model files found for filters: {not_found} at {outdir}"
            )

    def __repr__(self):
        return super().__repr__() + f"(model={self.model}, svd_path={self.svd_path})"

    def combine_lc_params(self, parameters):
        """Order the parameters as the surrogate expects them.

        The surrogate takes a positional list rather than a mapping, so the
        order of model_parameters is what matters here.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        list
            Parameter values, in the surrogate's own order.
        """
        return [
            parameters[key] if key in parameters else getattr(self, key)
            for key in self.model_parameters
        ]

    def generate_lightcurve(self, sample_times, parameters, filters="all"):
        """Evaluate the surrogate, in magnitudes or bolometrically.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.
        filters: list of str or str or None, optional
            Filters to evaluate. ``all`` uses the model's own list; None
            asks for the bolometric luminosity instead of magnitudes.

        Returns
        -------
        dict or numpy.ndarray
            Absolute magnitude per filter, or the bolometric luminosity.
        """
        parameters_list = self.em_parameter_setup(parameters)
        # parameters_list.append(param_dict["redshift"])

        # if filters is set to None, we assume we want the bolometric luminosity
        if filters is None:
            return lc_gen.calc_svd_lbol(
                sample_times,
                parameters_list,
                svd_lbol_model=self.svd_lbol_model,
                lbol_ncoeff=self.lbol_ncoeff,
            )  # original code provides a redshift correction of (1+z), but should this not rather be (1+z)**2

        elif filters == "all":
            filters = self.filters
        return lc_gen.calc_svd_lc(
            sample_times,
            parameters_list,
            self.svd_mag_model,
            mag_ncoeff=self.mag_ncoeff,
            filters=filters,
        )

    def generate_spectra(self, sample_times, wavelengths, parameters):
        """Evaluate the surrogate on wavelengths rather than on filters.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        wavelengths: list
            Wavelengths to evaluate, used in place of filter names.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Flux per wavelength.
        """
        return self.generate_lightcurve(sample_times, parameters, filters=wavelengths)


class FiestaKilonovaModel(FiestaModel):
    """A light curve model object for GRB light curves using fiesta

    An object to evaluate the GRB light curve across filters
    with a set of parameters given based on fiesta.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "Bu2026_MLP".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    surrogate_dir: str, optional
        path to the directory containing the surrogate models.
    **kwargs: optional
        Additional keyword arguments to be passed to the underlying fiesta model.

    Returns
    -------
    LightCurveModel: `nmma.em.model.GRBFiestaModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    load_dir_string = "KN"

    def __init__(self, model="Bu2026_MLP", **kwargs):
        super().__init__(model, **kwargs)


class GRBMixin:
    """Shared GRB behaviour, mixed into both the fiesta and afterglowpy models.

    Carries the jet geometry conversions and the checks that keep the
    afterglow physically meaningful.
    """

    def __init__(self, *args, resolution=12, **kwargs):
        self.resolution = resolution
        super().__init__(*args, **kwargs)

    def parameter_conversion(self, parameters):
        """Derive the jet angles and the total microphysical efficiency.

        Sampling the ratio of the wing to the core angle rather than the
        wing angle itself keeps the jet resolvable by construction, so the
        wing angle is reconstructed here when that ratio was sampled.

        Parameters
        ----------
        parameters: dict
            Sampled parameters, modified in place.

        Returns
        -------
        dict
            The parameters, with thetaWing and epsilon_tot added.
        """

        # it is beneficial to sample the ratio of angles alpha instead of checking later whether this can be resolved
        try:
            # FIXME We should get rid of the resolution attribute and only use alphaWing
            self.resolution = (
                parameters["alphaWing"] + 1e-10
            )  # to avoid precision issues
            parameters["thetaWing"] = parameters["alphaWing"] * parameters["thetaCore"]
        except KeyError:
            parameters["thetaWing"] = parameters["thetaWing"]

        try:
            parameters["epsilon_tot"] = (
                parameters["epsilon_e"] + parameters["epsilon_B"]
            )
        except KeyError:
            parameters["epsilon_tot"] = (
                10 ** parameters["log10_epsilon_e"]
                + 10 ** parameters["log10_epsilon_B"]
            )
        new_parameters = super().parameter_conversion(parameters)
        return new_parameters

    def sanity_checks(self, parameters):
        """Reject jet geometries the afterglow model cannot describe.

        A jet must fit in a hemisphere, must be wide enough to be resolved
        on the angular grid, must have a core large enough to be meaningful,
        and cannot put more energy into the microphysics than it has.

        Parameters
        ----------
        parameters: dict
            Parameters about to be evaluated.
        """
        # FIXME if ratio zeta and ratio epsilon are given,
        # we should check that the GRB energy does not
        # exceed the energy equivalent of the disk mass!
        self.good_parameters = (
            (parameters["thetaWing"] <= np.pi / 2)
            & (parameters["thetaWing"] / parameters["thetaCore"] <= self.resolution)
            & (parameters["thetaCore"] > np.pi / 1800.0)
            & (parameters["epsilon_tot"] <= 1.0)
        )


class FiestaGRBModel(GRBMixin, FiestaModel):
    """A light curve model object for GRB light curves using fiesta

    An object to evaluate the GRB light curve across filters
    with a set of parameters given based on fiesta.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "afgpy_gaussian_CVAE".
    filters: list of str, optional
        List of filters to create model for. Defaults to all available filters.
    surrogate_dir: str, optional
        path to the directory containing the surrogate models.
    **kwargs: optional
        Additional keyword arguments to be passed to the underlying Fiesta GRB model.

    Returns
    -------
    LightCurveModel: `nmma.em.model.GRBFiestaModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    load_dir_string = "GRB"

    def __init__(self, model="afgpy_gaussian_CVAE", **kwargs):
        super().__init__(model, **kwargs)


class GRBLightCurveModel(GRBMixin, LightCurveModelContainer):
    """A light curve model object for GRB light curves using afterglowpy

    An object to evaluate the GRB light curve across filters
    with a set of parameters given based on afterglowpy.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "TrPi2018".
    resolution: int, optional
        Resolution for the GRB model. Default is 12.
    jet_type: int, optional
        Type of jet for the GRB model. Default is 0.
    **kwargs: optional
        Additional keyword arguments to be passed to the LightCurveModelContainer or underlying GRB model.

    Returns
    -------
    LightCurveModel: `nmma.em.model.GRBLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    def __init__(
        self,
        model="TrPi2018",
        resolution=12,
        jet_type=0,
        **kwargs,
    ):
        super().__init__(model, resolution=resolution, **kwargs)
        self.jet_type = jet_type
        self.default_parameters = {
            "xi_N": 1.0,
            "d_L": 3.086e19,  # d_L=10pc in cm
            "jetType": jet_type,
            "specType": 0,
            **self.kwargs,
        }
        self.def_keys = self.default_parameters.keys()
        # keys we typically sample in log space, but need to convert to linear space
        self.log_sampling_keys = ["E0", "n0", "epsilon_e", "epsilon_B"]
        self.energy_injection_params = [
            "energy_exponential",
            "log10_Eend",
            "t_start",
            "injection_duration",
        ]
        self.flux_func = None  # flux function to be set later, if needed

    def setup_model_times(self):
        """Use a grid spanning the whole afterglow, from seconds to months.

        Returns
        -------
        numpy.ndarray
            Times from 1e-5 to 200 days.
        """
        tmin = 1.0e-5  # minimum time in days
        tmax = 200  # maximum time in days
        nsteps = 201  # number of time steps
        return np.geomspace(tmin, tmax, nsteps)

    def em_parameter_setup(self, parameters):
        """Pick the afterglow evaluator, then cache the frame quantities.

        Whether the jet is energised over time or carries a fixed energy is
        decided once, on the first call, from the parameters present.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Parameters in the form afterglowpy expects.
        """

        # set on first call
        if self.flux_func is None:
            # case 1: use energy injection approach
            if all(key in parameters for key in self.energy_injection_params):
                self.flux_func = lc_gen.flux_density_on_E0_array
                self.log_sampling_keys.remove("E0")
            else:  # case 2
                self.flux_func = lc_gen.flux_density_on_time_array

        super().em_parameter_setup(parameters, combine_params=False)

        # set the default parameters, preferentially from sampling
        grb_param_dict = {
            k: parameters.get(k, self.default_parameters[k]) for k in self.def_keys
        }

        grb_param_dict["z"] = self.redshift
        grb_param_dict["thetaObs"] = parameters[
            "inclination_EM"
        ]  # afterglowpy uses viewing angle in rad!

        for key in self.log_sampling_keys:
            try:
                grb_param_dict[key] = parameters[key]
            except KeyError:
                grb_param_dict[key] = 10 ** parameters[f"log10_{key}"]

        # make sure L0, q and ts are also passed
        for param in ["thetaCore", "thetaWing", "p", "L0", "q", "ts"]:
            try:
                grb_param_dict[param] = parameters[param]
            except KeyError:
                pass

        if grb_param_dict["jetType"] in [1, 4]:
            grb_param_dict["b"] = parameters["b"]
        return grb_param_dict

    def generate_lightcurve(self, sample_times, parameters):
        """Evaluate the GRB afterglow.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Absolute magnitude per filter, or an empty dict when the jet
            geometry was rejected.
        """
        grb_param_dict = self.em_parameter_setup(parameters)

        # sanity checks
        if not self.good_parameters:
            return {}
        else:
            return lc_gen.afterglowpy_lc(
                sample_times,
                grb_param_dict,
                filters=self.default_filts,
                obs_frequencies=self.nu_0s,
                flux_func=self.flux_func,
            )


class HostGalaxyLightCurveModel(LightCurveModelContainer):
    """
    A light curve model object for a simple lightcurve model that includes the transient's host galaxy.

    An object to evaluate the host galaxy light curve across filters
    with a set of parameters given based on arxiv:2303.12849.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "Sr2023".
    host_mag: float or int, optional
        Magnitude of the host galaxy. Default is 23.9.
    **kwargs: optional
        Additional keyword arguments for LightCurveModelContainer.

    Returns
    -------
    LightCurveModel: `nmma.em.model.HostGalaxyLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    def __init__(self, model="Sr2023", host_mag=23.9, **kwargs):
        super().__init__(model, **kwargs)
        if isinstance(host_mag, (float, int)):
            host_mag = np.full_like(self.filters, host_mag)
        self.host_mag = host_mag

    def check_vs_priors(self, priors):
        """Refuse an extinction prior, the host model carrying its own.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.

        Raises
        ------
        ValueError
            If Ebv is among the priors.
        """
        super().check_vs_priors(priors)
        if "Ebv" in priors:
            raise ValueError(
                "Host galaxy model does not support additional extinction correction."
                "This should be part of the model. Please remove Ebv from priors."
            )

    def generate_lightcurve(self, sample_times, parameters):
        """Evaluate the host galaxy and its fading afterglow.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Absolute magnitude per filter.
        """
        parameters = self.em_parameter_setup(parameters)
        return lc_gen.host_lc(sample_times, parameters, self.filters, self.host_mag)


class SupernovaLightCurveModel(LightCurveModelContainer):
    """
    A light curve model object for supernova light curves using sncosmo

    An object to evaluate the supernova light curve across filters
    with a set of parameters given based on sncosmo.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "nugent-hyper".
    **kwargs: optional
        Additional keyword arguments to be passed to the LightCurveModelContainer or to the underlying sncosmo model.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SupernovaLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    def __init__(self, model="nugent-hyper", **kwargs):
        if isinstance(model, str):
            self.source = sncosmo.get_source(model)
            model_name = self.source.name
        else:
            self.sn_model = model
            self.source = self.sn_model._source
            model_name = self.sn_model._source.name
        super().__init__(model_name, **kwargs)

        if isinstance(model, str):
            self.sn_model = sncosmo.Model(source=model, **self.kwargs)

        self.model_parameters = self.sn_model.param_names

    def identify_model_parameters(self, model_parameters):
        """Take the parameter names from sncosmo rather than from NMMA.

        Parameters
        ----------
        model_parameters: None
            Must be None; sncosmo owns the parameter list here.

        Raises
        ------
        ValueError
            If model parameters are supplied anyway.
        """

        if model_parameters is not None:
            raise ValueError(
                "Model_parameters are ignored for SupernovaLightCurveModel, using sncosmo defaults."
            )

    def setup_model_times(self):
        """Use the phase range the sncosmo source covers.

        Some sncosmo models are defined relative to peak brightness and some
        relative to the explosion. NMMA always counts from the explosion, so
        a model starting at negative phase is shifted and the user warned,
        their t0 prior needing to follow.

        Returns
        -------
        numpy.ndarray
            Source-frame times, in days.
        """
        # NOTE:we want these in source frame:
        sample_times = np.linspace(self.source.minphase(), self.source.maxphase(), 200)

        if (sample_times < 0).any():
            # NOTE: We assume this means the sncosmo model is relative to peak time.
            sample_times -= sample_times[0]
            print(
                "Warning: Some supernova models are relative to the peak, some relative to the explosion time, "
                "but nmma always expects times relative to the explosion time. Adjust your t0 prior accordingly."
            )
        return sample_times

    def check_vs_priors(self, priors):
        """Explain how NMMA parametrises a supernova, then check the priors.

        sncosmo sources are normalised by an amplitude whose scale differs
        wildly between models. NMMA offers an alternative anchored on a
        fiducial absolute magnitude, which is what the printed note is
        about.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.
        """
        print(
            """
            Note: Most source models in sncosmo use an 'amplitude' parameter,
            that can differ drastically from model to model and requires a carefully chosen prior.
            NMMA allows an alternative approach that anchors the peak magnitude
            with a fiducial absolute magnitude of -19.35 and samples the parameters
            'supernova_mag_boost' and 'supernova_mag_stretch' instead to allow a more
            direct phyiscal interpretation.
            Your sampling priors should be set accordingly. See the NMMA documentation for more details.
        """
        )

        if "supernova_mag_boost" in priors or "supernova_mag_stretch" in priors:
            self._anchor_amplitude()
            if "amplitude" in priors or "salt" in self.model:
                raise ValueError(
                    """You cannot sample the sncosmo amplitude parameter and the 'supernova_mag_boost' or 'supernova_mag_stretch' parameters at the same time because they contain identical information. """
                )

        else:
            self.get_lc = self.sncosmo_lc

        super().check_vs_priors(priors)

    def _anchor_amplitude(self, mag_ref=-19.35):
        """Pin the source rest-frame peak magnitude to mag_ref.

        Tries reference bands in order of preference; a band fails with
        ValueError only if it lies outside the template's wavelength
        coverage. The anchor value is band-dependent (colour term), so the
        chosen band is reported once; keep supernova_mag_boost sampled to
        absorb this.
        """
        self.mag_ref = mag_ref
        for band, magsys in (
            ("bessellv", "vega"),
            ("bessellb", "vega"),
            ("sdssg", "ab"),
        ):
            try:
                self.sn_model.set_source_peakmag(mag_ref, band, magsys)
                print(
                    f"SupernovaLightCurveModel ({self.model}): source peak "
                    f"anchored to M = {mag_ref} in '{band}' ({magsys})."
                )
                return
            except ValueError:
                continue
        raise RuntimeError(
            f"SupernovaLightCurveModel ({self.model}): amplitude anchoring failed for all "
            f"candidate bands (model wavelength range "
            f"{self.sn_model.minwave():.0f}-{self.sn_model.maxwave():.0f} A). "
            "Use a template with optical coverage (bessellv, bessellb, or sdssg)."
        )

    def em_parameter_setup(self, parameters):
        """Push the parameters into the sncosmo model itself.

        sncosmo keeps its own state, so there is nothing to return: the
        model is configured in place.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.
        """
        lc_pars = super().em_parameter_setup(parameters)
        self.sn_model.set(**lc_pars)

    def combine_lc_params(self, parameters):
        """Fill in the parameters sncosmo needs, keeping its own defaults.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model, modified in place.

        Returns
        -------
        dict
            Parameters in sncosmo's own naming, redshift included.
        """
        parameters["t0"] = parameters.get("t0", 0.0)
        parameters["z"] = self.redshift
        return {
            p: parameters.get(p, self.sn_model.get(p)) for p in self.model_parameters
        }

    def gen_detector_lc(self, parameters=None, sample_times=None):
        """Generate the observer-frame light curve of the supernova.

        Parameters
        ----------
        parameters: dict, optional
            Parameters of the supernova model.
        sample_times: numpy.ndarray, optional
            Source-frame times to evaluate. Defaults to the model's grid.

        Returns
        -------
        tuple
            Observer-frame times in days, and the apparent magnitude per
            filter.
        """

        self.em_parameter_setup(parameters)
        if sample_times is None:
            sample_times = self.model_times

        # expected by snmodel in obs_frame, readjusted internally
        sample_times_obs = sample_times * (1.0 + self.redshift)
        mags = self.get_lc(sample_times_obs, parameters)

        if self.extinction_frame:
            mags = self.extinction_correction(mags)
        return (sample_times_obs + self.timeshift, mags)

    def get_lc(self, sample_times_obs, parameters):
        """Evaluate the supernova, through NMMA's own parametrisation.

        Parameters
        ----------
        sample_times_obs: numpy.ndarray
            Observer-frame times, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Apparent magnitude per filter.
        """
        return self.nmma_lc(sample_times_obs, parameters)

    def nmma_lc(self, sample_times_obs, parameters):
        """Evaluate the supernova with a stretched and shifted light curve.

        Rather than sncosmo's amplitude, this scales the timescale and
        offsets the magnitude, which have a more direct physical reading.

        Parameters
        ----------
        sample_times_obs: numpy.ndarray
            Observer-frame times, in days.
        parameters: dict
            Holds supernova_mag_stretch and supernova_mag_boost.

        Returns
        -------
        dict
            Apparent magnitude per filter.
        """
        stretch = (parameters.get("supernova_mag_stretch", 1.0),)
        mags = self.sncosmo_lc(sample_times_obs / stretch)

        mag_boost = parameters.get("supernova_mag_boost", 0.0)
        for filt in mags:
            mags[filt] = np.asarray(mags[filt]) + mag_boost + self.distmod
        return mags

    def sncosmo_lc(self, sample_times_obs, parameters=None):
        """Read magnitudes straight off the sncosmo model.

        Parameters
        ----------
        sample_times_obs: numpy.ndarray
            Observer-frame times, in days.
        parameters: dict, optional
            Unused; the model has already been configured.

        Returns
        -------
        dict
            Absolute magnitude per filter.
        """
        return lc_gen.sn_lc(
            sample_times_obs,
            self.sn_model,
            self.default_filts,
            self.lambdas,
        )


class ShockCoolingLightCurveModel(LightCurveModelContainer):
    """Light curve of an envelope cooling after the shock breaks out.

    Describes the first hours to days of a transient, before any
    radioactive heating takes over.
    """

    def __init__(self, model="Piro2021", **kwargs):
        """A light curve model object

                An object to evaluted the shock cooling light curve across filters, particularly suited for descriptions of lightcurves at early times (hours to few days)

            Parameters
            ----------
        model: str, optional
            Name of the model. Default is "Piro2021".
        **kwargs: optional
            Additional keyword arguments for the LightCurveModelContainer.

                Returns
                -------
                LightCurveModel: `nmma.em.model.ShockCoolingLightCurveModel`
                    A light curve model object to evaluate the light curve
                    from a set of parameters
        """
        super().__init__(model, **kwargs)

    def setup_model_times(self):
        """Use a grid covering the hours-to-days window the model describes.

        Returns
        -------
        numpy.ndarray
            Times from one hour to 3.5 days.
        """
        # model is suitable on the order of hours to a few days
        # this limit is somewhat arbitrary, but should be sufficient for most cases
        return np.geomspace(1.0 / 24.0, 3.5, 100)

    def generate_lightcurve(self, sample_times, parameters, filters="all"):
        """Evaluate the shock cooling model.

        The bolometric luminosity and the photospheric radius are computed
        together, then turned into magnitudes through a blackbody.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.
        filters: list of str or str, optional
            Filters to evaluate. ``all`` uses the model's own list; None
            asks for the bolometric luminosity instead.

        Returns
        -------
        dict or numpy.ndarray
            Absolute magnitude per filter, or the bolometric luminosity.
        """
        lc_param_dict = self.em_parameter_setup(parameters)

        if filters is None:
            return lc_gen.sc_bol_lc(sample_times, lc_param_dict, compute_Rs=False)
            # would need a (1+z)**2 correction in observ. frame luminosity

        elif filters == "all":
            filters = self.default_filts
        lbol, Rs = lc_gen.sc_bol_lc(sample_times, lc_param_dict, compute_Rs=True)

        nu_host = self.nu_0s * (1 + self.redshift)  # convert to host frame
        return lc_gen.sc_lc(lbol, Rs, nu_host, filters)


class SimpleKilonovaLightCurveModel(LightCurveModelContainer):
    """
    A light curve model object for simple kilonova light curves

    An object to evaluate the kilonova light curve across filters
    with a set of parameters given based on simple analytical models.

    Parameters
    ----------
    model: str, optional
        Name of the model. Default is "Me2017".
    **kwargs: optional
        Additional keyword arguments for the LightCurveModelContainer.

    Returns
    -------
    LightCurveModel: `nmma.em.model.SimpleKilonovaLightCurveModel`
        A light curve model object to evaluate the light curve
        from a set of parameters.
    """

    lc_dict = {
        # FIXME weizmann: to be activated separately, after the NMMA
        # documentation work.
        # "HoNa2020": lc_gen.HoNa_lc,
        "HoHa2020": lc_gen.HoNa_lc,
        "Me2017": lc_gen.eff_metzger_lc,
        "PL_BB_fixedT": lc_gen.powerlaw_blackbody_constant_temperature_lc,
        "blackbody_fixedT": lc_gen.blackbody_constant_temperature,
        "synchrotron_powerlaw": lc_gen.synchrotron_powerlaw,
    }

    def __init__(self, model="Me2017", **kwargs):
        super().__init__(model, **kwargs)
        self.lc_func = self.lc_dict[model]

        if model in ["HoNa2020", "Me2017"] and np.min(self.model_times) < 5e-2:
            print("Warning: chosen KN model is only valid for times >= 0.05 days.")
            print("Setting minimum time above 0.05 days.")
            self.model_times = self.model_times[self.model_times >= 5e-2]

    def generate_lightcurve(self, sample_times, parameters):
        """Evaluate the chosen semi-analytical kilonova model.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.

        Returns
        -------
        dict
            Absolute magnitude per filter.
        """
        param_dict = self.em_parameter_setup(parameters)
        param_dict["distance_modulus"] = self.distmod

        nu_host = self.nu_0s * (1 + self.redshift)  # convert to host frame
        # prevent the output message flooded by these warning messages
        old = np.seterr()
        np.seterr(invalid="ignore")
        np.seterr(divide="ignore")
        mag = self.lc_func(sample_times, param_dict, nu_host, self.default_filts)
        np.seterr(**old)

        return mag


class CombinedLightCurveModelContainer(LightCurveModelContainer):
    """
    An object to evaluate the combined light curve from a set of parameters
    using multiple light curve models.

    Parameters
    ----------
    models: list
        A list of LightCurveModel instances.
    model_args: list, optional
        If not None (default), this is assumed to be a list of argument lists
        with which the respective submodels will be initialized.

    Returns
    -------
    LightCurveModel: `nmma.em.CombinedLightCurveModelContainer`
        A light curve model object to evaluate the combined
        light curve from a set of parameters.
    """

    def __init__(self, models, model_args=None):
        if model_args is None:
            self.lc_models = models
        else:
            self.lc_models = [model(*model_args[i]) for i, model in enumerate(models)]
        self.model = [lc_model.model for lc_model in self.lc_models]

        self.all_filters = set().union(*[model.filters for model in self.lc_models])
        # FIXME: Better treatment for synonymous or equivalent filters?
        self.compatible_filters, _ = utils.get_filter_name_mapping(self.all_filters)

        self.model_times = np.array(
            sorted(set().union(*[model.model_times for model in self.lc_models]))
        )

    def __repr__(self):
        if len(self.lc_models) == 2:
            sub_str = (
                f"{self.lc_models[0].__repr__()} and {self.lc_models[1].__repr__()}"
            )
        else:
            sub_str = f"{self.__class__.__name__} of {', '.join([lc_model.__repr__() for lc_model in self.lc_models[:-1]])} and {self.lc_models[-1].__repr__()}"
        return f"Combination of {sub_str}"

    def check_vs_priors(self, priors):
        """Let every component check the priors in turn.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.
        """
        for model in self.lc_models:
            model.check_vs_priors(priors)

    @property
    def citation(self):
        """The references of every component model.

        Returns
        -------
        dict
            Each model name, mapped to its references.
        """
        citations = {}
        for lc_model in self.lc_models:
            citations.update(lc_model.citation)
        return citations

    @property
    def good_parameters(self):
        """Whether every component accepted the current parameters.

        Returns
        -------
        bool
            False as soon as one component rejected them.
        """
        return np.prod([lc_model.good_parameters for lc_model in self.lc_models])

    @good_parameters.setter
    def good_parameters(self, value):
        """Set the flag on every component at once.

        Parameters
        ----------
        value: bool
            Value to propagate.
        """
        for lc_model in self.lc_models:
            lc_model.good_parameters = value

    def parameter_conversion(self, parameters):
        """Chain the conversions of every component.

        Each model sees what the previous ones added, so a quantity derived
        by one is available to the next.

        Parameters
        ----------
        parameters: dict
            Sampled parameters.

        Returns
        -------
        dict
            The parameters, after every component has had its turn.
        """
        for lc_model in self.lc_models:
            parameters = lc_model.parameter_conversion(parameters)
        return parameters

    def gen_detector_lc(self, parameters, sample_times=None, return_all=False):
        """Generate the observer-frame light curve of every component.

        Each component handles its own frame conversion, then the results
        are interpolated onto a common time grid and stacked. If any one
        component rejects the parameters, its empty result is returned
        straight away.

        Parameters
        ----------
        parameters: dict
            Parameters of the light curve model.
        sample_times: numpy.ndarray, optional
            Source-frame times to evaluate. Each model uses its own grid
            when omitted.
        return_all: bool, optional
            If True, return the components separately instead of stacked.

        Returns
        -------
        tuple
            Observer-frame times in days, and the apparent magnitude per
            filter.
        """

        lc_per_model = []
        time_per_model = []
        for model in self.lc_models:
            ref_times, lc = model.gen_detector_lc(parameters, sample_times)

            if not lc:
                # if the model returns False/empty, it means that the model
                # could not generate a light curve for the given parameters
                return (ref_times, lc)
            else:
                lc_per_model.append(lc)
                time_per_model.append(ref_times)

        if return_all:
            return (time_per_model, lc_per_model)
        if sample_times is None:
            connected_times = np.array(sorted(set().union(*time_per_model)))
        else:
            connected_times = ref_times  # if we have sample_times, the ref_times are the same for all models
        if isinstance(lc, dict):
            # if the (last) model returns a dictionary, we have light curves per filter

            lcs_on_joint_time_frame = [
                {
                    filt: utils.autocomplete_data(
                        connected_times, ref_times, filt_lc, extrapolate=np.inf
                    )
                    for filt, filt_lc in lc.items()
                }
                for ref_times, lc in zip(time_per_model, lc_per_model)
            ]
            total_lc = self.stack_magnitudes(lcs_on_joint_time_frame)

        elif isinstance(lc, (list, np.ndarray)):
            # if the model returns array-like, this was a bolometric light curve
            lcs_on_joint_time_frame = [
                utils.autocomplete_data(connected_times, ref_times, lc, extrapolate=0.0)
                for ref_times, lc in zip(time_per_model, lc_per_model)
            ]
            total_lc = np.sum(lcs_on_joint_time_frame, axis=0)

        return connected_times, total_lc

    def generate_lightcurve(self, sample_times, parameters, return_all=False):
        """Evaluate every component and add up their fluxes.

        Parameters
        ----------
        sample_times: numpy.ndarray
            Source-frame times to evaluate, in days.
        parameters: dict
            Parameters of the light curve model.
        return_all: bool, optional
            If True, return the components separately instead of stacked.

        Returns
        -------
        dict or list
            The combined magnitude per filter, or one dict per component.
        """

        lc_per_model = []
        for model in self.lc_models:
            lc = model.generate_lightcurve(sample_times, parameters)

            if not lc:
                # if the model returns False/empty, it means that the model
                # could not generate a light curve for the given parameters
                return lc
            else:
                lc_per_model.append(lc)

        if return_all:
            return lc_per_model

        if isinstance(lc, (list, np.ndarray)):
            # if the model returns array-like, this was a bolometric light curve
            total_lc = np.sum(lc_per_model, axis=0)

        elif isinstance(lc, dict):
            total_lc = self.stack_magnitudes(lc_per_model)

        return total_lc

    def stack_magnitudes(self, mags_per_model):
        """Add up several models into one light curve.

        Magnitudes are logarithmic, so they cannot simply be summed: each is
        converted to a flux, the fluxes are added, and the total is turned
        back into a magnitude.

        Parameters
        ----------
        mags_per_model: list of dict
            Magnitude per filter, one dict per component model.

        Returns
        -------
        dict
            Combined magnitude per filter.
        """
        stacked_mags = {}
        for filt in self.all_filters:
            mAB_list = []
            for mag in mags_per_model:
                try:
                    mag_per_filt = mag[self.compatible_filters[filt]]
                except KeyError:
                    # if the filter is not available in the model,
                    # the submodels may still be compatible
                    try:
                        mag_per_filt = utils.average_mags(mag, filt)
                    except ValueError:
                        continue
                mAB_list.append(-2.0 / 5.0 * ln10 * np.array(mag_per_filt))
            if not mAB_list:
                # return no magnitude if no good value in either model
                stacked_mags[filt] = np.full_like(self.model_times, np.inf)
            else:
                stacked_mags[filt] = -5.0 / 2.0 * logsumexp(mAB_list, axis=0) / ln10
        return stacked_mags


def single_model_from_args(
    model_class, model_name, args, filters, prefixes=["grb_", "em_"]
):
    """Build one model, taking its arguments from the command line.

    The constructor signature is inspected so that only the arguments the
    model actually accepts are passed, which is what lets very different
    models share one command line.

    Parameters
    ----------
    model_class: type
        Class to instantiate.
    model_name: str or None
        Name of the model, stripped of surrounding whitespace.
    args: argparse.Namespace
        Parsed command-line arguments.
    filters: list of str
        Filters the model is built for.
    prefixes: list of str, optional
        Argument prefixes to strip when matching names.

    Returns
    -------
    LightCurveModelContainer
        The instantiated model.
    """

    # populate model-args from default and parsed args
    model_args = initialisation_args_from_signature_and_namespace(
        LightCurveModelContainer, args, prefixes=prefixes
    )
    model_args |= initialisation_args_from_signature_and_namespace(
        model_class, args, prefixes=prefixes
    )
    # update explicit args
    model_args |= dict(
        filters=filters,
        sample_times=utils.setup_sample_times(args),
    )
    if model_name is not None:
        model_args["model"] = model_name.strip()
    if args.em_model_kwargs:
        model_args |= args.em_model_kwargs

    return model_class(**model_args)


def create_light_curve_model_from_args(args, filters=None):
    """Build the light curve model an analysis will use.

    Several transient classes can be requested at once, in which case they
    are combined into a single model.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments.
    filters: list of str, optional
        Filters to build for. Derived from args when omitted.

    Returns
    -------
    LightCurveModelContainer
        One model, or a combined container when several were requested.
    """
    if filters is None:
        filters = utils.set_filters(args)
    if getattr(args, "em_transient_class", None):
        if isinstance(args.em_transient_class, str):
            transient_classes = args.em_transient_class.split(",")
        else:
            transient_classes = list(args.em_transient_class)
        em_model = getattr(args, "em_model", None)
        model_classes, model_names = lc_model_from_transient_class(
            transient_classes, em_model
        )

    else:
        model_names = args.em_model
        if isinstance(model_names, str):
            model_names = model_names.split(",")
        else:
            model_names = list(model_names)
        model_classes = [single_model_from_mapping(mn) for mn in model_names]

    models_list = [
        single_model_from_args(mc, mn, args, filters)
        for mc, mn in zip(model_classes, model_names)
    ]
    if len(models_list) == 1:
        return models_list[0]
    print("Running with combination of multiple light curve models")
    return CombinedLightCurveModelContainer(models_list)


def lc_model_from_transient_class(transient_class, em_model):
    """Pair each requested transient class with the model name it should use.

    Classes and names are given as two separate lists on the command line
    and are matched by position. A class left without a name falls back to
    its own default model.

    Parameters
    ----------
    transient_class: list of str
        Transient classes requested.
    em_model: str or list or None
        Model names, comma-separated or already a list.

    Returns
    -------
    tuple
        The model classes, and the model name for each of them.

    Raises
    ------
    ValueError
        If em_model is neither a string, a list, nor None.
    """
    model_classes = [
        single_model_from_mapping(tc, enfore_class=True) for tc in transient_class
    ]

    if isinstance(em_model, str):
        prel_model_names = em_model.split(",")
    elif isinstance(em_model, list):
        prel_model_names = em_model
    elif em_model is None:
        prel_model_names = []
    else:
        raise ValueError(f" {em_model} is not a valid input for em_model.")

    model_names = []
    for i, model_class in enumerate(model_classes):
        try:
            model_names.append(prel_model_names[i])
        except IndexError:
            print(
                f"Warning: No model name found for {model_class}. Will try using the default model"
            )
            model_names.append(None)
    return model_classes, model_names


def single_model_from_mapping(identifier, enfore_class=False):
    """Resolve a transient class name into the class that implements it.

    Parameters
    ----------
    identifier: str or LightCurveModelContainer
        Name of the transient class, or an already-built model, which is
        returned unchanged.
    enfore_class: bool, optional
        If True, require the name to be a known transient class rather than
        falling back on the surrogate model.

    Returns
    -------
    type or LightCurveModelContainer
        The model class, or the model that was passed in.
    """
    if isinstance(identifier, LightCurveModelContainer):
        return identifier

    identifier = identifier.strip()

    transient_class_map = {
        "svd": SVDLightCurveModel,
        "fiesta_kn": FiestaKilonovaModel,
        "fiesta_grb": FiestaGRBModel,
        "grb": GRBLightCurveModel,
        "host_galaxy": HostGalaxyLightCurveModel,
        "supernova": SupernovaLightCurveModel,
        "shock": ShockCoolingLightCurveModel,
        "simple_kilonova": SimpleKilonovaLightCurveModel,
    }
    if identifier.casefold() in transient_class_map.keys():
        return transient_class_map[identifier.casefold()]
    elif enfore_class:
        raise ValueError(
            f"Transient class {identifier} not recognized. Please choose from {list(transient_class_map.keys())}"
        )

    # FIXME This is incomplete, but identical to handling in NMMA 0.2.2
    model_name_map = {
        "TrPi2018": GRBLightCurveModel,
        "Piro2021": ShockCoolingLightCurveModel,
        "Me2017": SimpleKilonovaLightCurveModel,
        "PL_BB_fixedT": SimpleKilonovaLightCurveModel,
        "Sr2023": HostGalaxyLightCurveModel,
        "Arnett": SimpleBolometricLightCurveModel,
    }
    if identifier in model_name_map.keys():
        return model_name_map[identifier]
    elif identifier in [val["name"] for val in _SOURCES.get_loaders_metadata()]:
        return SupernovaLightCurveModel
    else:
        # FIXME This is an unclean default, should be more explicit!
        return SVDLightCurveModel


def create_injection_model(args, filters=None):
    """Build the model used to simulate an injection.

    The injection reuses the analysis arguments, except where an
    injection-specific one was given: those override their counterparts, so
    that an injection can deliberately differ from the model being fitted.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments.
    filters: list of str, optional
        Filters to build for.

    Returns
    -------
    LightCurveModelContainer
        The model to generate the injection with.
    """
    # By default, injection takes same args as the main model
    injection_args = copy(args)

    # replace injection-specific args
    for arg, val in args.__dict__.items():
        if arg == "injection_model_args":
            if val is None:
                injection_dict = {}
            elif isinstance(val, str):
                injection_dict = literal_eval(val)
            else:
                injection_dict = val
            for arg, val in injection_dict.items():
                arg = arg.lstrip("--").replace("-", "_").replace("injection_", "")
                setattr(injection_args, arg, val)
        elif arg.startswith("injection_"):  # replace 'injection_' prefix if necessary
            setattr(injection_args, arg.replace("injection_", ""), val)

    return create_light_curve_model_from_args(injection_args, filters)
