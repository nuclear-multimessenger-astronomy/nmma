from ast import literal_eval
import inspect
import numpy as np
from bilby.core import prior as bprior
import warnings
from ..core.utils import load_yaml
from .utils import autocomplete_data, set_filter_associated_dict


class ValidationError(ValueError):
    """Raised when a systematics configuration file is malformed.

    Parameters
    ----------
    key: str
        Configuration entry the problem was found in.
    message: str
        What is wrong with it.
    """

    def __init__(self, key, message):
        super().__init__(f"Validation error for '{key}': {message}")


class SystematicsHandler:
    """Model the uncertainty the light curve model itself carries.

    Measurement errors are not the only source of scatter: the model is an
    approximation, and how wrong it is varies with time. This turns a YAML
    description of that budget into priors the sampler can explore, then
    evaluates it at each step.

    Three modes are supported, in increasing order of freedom: a fixed
    budget that is not sampled at all, a single sampled value, and a set of
    values on a time grid that are interpolated between.
    """

    allowed_keys = ["time_range", "time_nodes", "prior", "params", "each", "filters"]

    def __init__(
        self,
        systematics_file=None,
        error_budget=None,
        light_curve_times=np.linspace(0.1, 14, 10),
        base_prior_name="em_syserr",
    ):
        """
        systematics_file: str or dict (default: None)
            YAML file or dictionary defining the systematic uncertainties and their priors.
        error_budget: Any (default:1)
            Additionally introduced statistical error on the light curve data,
            so as to keep the systematic error under control. This will only be used if the parameters-dict does not contain base_prior_name.
        lc_times: np.ndarray (default: np.linspace(0.1, 14, 10))
            Times at which the light curve is evaluated, needed to set up the error budget.
        base_prior_name: str (default: "em_syserr")
            Base name of the systematic uncertainty prior(s).
        """
        self.base_prior_name = base_prior_name
        self.default_t_grid_type = "linear"
        self.light_curve_times = light_curve_times

        # preliminary error budget setup
        self.adjust_error_budget(error_budget)
        self.compute_em_err = self.from_budget

        if isinstance(systematics_file, str):
            self.systematics_dict = load_yaml(systematics_file)
        elif isinstance(systematics_file, dict):
            self.systematics_dict = systematics_file
        else:
            self.systematics_dict = {}

    def adjust_error_budget(self, error_budget):
        """Spread a scalar budget over the light curve times.

        Parameters
        ----------
        error_budget: float or str or None
            Budget in magnitudes. A negligible default is used when None.
        """
        if error_budget is None:
            error_budget = 0.0001
        if isinstance(error_budget, str):
            error_budget = float(error_budget)
        self.error_budget = np.full_like(self.light_curve_times, error_budget)

    def from_budget(self, _):
        """Return the fixed budget, ignoring the parameters.

        Parameters
        ----------
        _: dict
            Sampled parameters, unused in this mode.

        Returns
        -------
        numpy.ndarray
            The budget, one value per observing time.
        """
        return self.error_budget

    def __call__(self, parameters):
        return self.compute_em_err(parameters)

    def setup_systematics_priors(self, prior_dict):
        """Turn the YAML description into priors, and add them to the set.

        The structure follows the one used by fiesta. A ``config`` entry
        routes to the legacy reader instead.

        Parameters
        ----------
        prior_dict: bilby.core.prior.PriorDict
            Priors to extend, modified in place.

        Returns
        -------
        bilby.core.prior.PriorDict
            The priors, with the systematics ones added.
        """
        for key, info_dict in self.systematics_dict.items():
            if key == "config":
                return self.legacy_prior_setup(prior_dict)

            # case 0: only one global systematic uncertainty
            if key in self.allowed_keys:
                print(
                    "Single global systematic uncertainty setup detected, applying to all filters."
                )
                new_prior = self.setup_filt_prior("", self.systematics_dict)
                prior_dict.update(new_prior)
                return prior_dict

            new_priors = self.setup_filt_prior(key, info_dict)
            # case 1 (&2): create priors for each filter (or keep key if not specified)
            for filt in info_dict.get("each", [key]):
                filt_priors = {}
                for k, v in new_priors.copy().items():
                    new_key = k.replace(key, filt)
                    v.name = new_key
                    filt_priors[new_key] = v
                prior_dict.update(filt_priors)

        return prior_dict

    def setup_filt_prior(self, key, info_dict):
        """Build the prior or priors described by one YAML entry.

        An entry that is just a number becomes a fixed value; one asking for
        two or more time nodes becomes one prior per node; anything else
        becomes a single sampled prior.

        Parameters
        ----------
        key: str
            Name of the entry, used to name the priors.
        info_dict: dict or float
            The entry itself.

        Returns
        -------
        dict
            Priors, keyed by their name.
        """
        prior_name = self.prior_name(key)

        # case 0: just a constant
        if isinstance(info_dict, float):
            return {prior_name: bprior.DeltaFunction(name=prior_name, value=info_dict)}

        # case 1: time-dependent systematic uncertainty
        num = info_dict.get("time_nodes", info_dict.get("time_range", "1").split()[-1])
        if int(num) >= 2:
            prior_names = [f"{prior_name}_{i}" for i in range(int(num))]
            return {n: self.get_prior(info_dict, n) for n in prior_names}

        # case 2: time-independent systematic uncertainty
        return {prior_name: self.get_prior(info_dict, prior_name)}

    def get_prior(self, info_dict, n):
        """Instantiate one bilby prior from its textual description.

        The description is written as a call, ``Uniform(0, 1)``, and is
        either parsed by bilby or rebuilt from an explicit params entry.

        Parameters
        ----------
        info_dict: dict
            Entry holding the prior string and, optionally, its parameters.
        n: str
            Name to give the prior.

        Returns
        -------
        bilby.core.prior.Prior
            The instantiated prior.
        """
        prior_str = info_dict["prior"]
        cls = prior_str.split("(")[0]
        args = "(".join(prior_str.split("(")[1:])[:-1]
        prior_class = getattr(bprior, cls)
        if args:
            new_prior = prior_class.from_repr(args)
            new_prior.name = n
            return new_prior
        else:
            return prior_class(
                **info_dict.get("params", {}), **info_dict.get("kwargs", {}), name=n
            )

    def get_name_and_times(self, key, info_dict):
        """Resolve an entry into a prior name and its time grid.

        Parameters
        ----------
        key: str
            Name of the entry.
        info_dict: dict
            The entry itself.

        Returns
        -------
        tuple
            The prior name, and the time nodes, or None when the budget does
            not vary with time.
        """
        prior_name = self.prior_name(key)
        time_range = self.get_time_range(info_dict)
        return prior_name, time_range

    def prior_name(self, key):
        """Name a prior after its entry, or use the base name alone.

        Parameters
        ----------
        key: str
            Name of the entry, possibly empty.

        Returns
        -------
        str
            The prior name.
        """
        return f"{self.base_prior_name}_{key}" if key else self.base_prior_name

    def get_time_range(self, info_dict):
        """Build the time grid the budget is sampled on.

        The range may be written in several shortened forms, from a bare
        node count to a full ``log 0.1 14 5``. Anything missing falls back
        to the span of the light curve and to a linear grid.

        Parameters
        ----------
        info_dict: dict
            Entry holding time_nodes or time_range.

        Returns
        -------
        numpy.ndarray or None
            The time nodes, or None when the budget does not vary with time.

        Raises
        ------
        ValueError
            If the time range cannot be read.
        """
        num = info_dict.get("time_nodes", None)
        t_range = info_dict.get("time_range", "").split()
        if num is None and t_range:
            num = t_range.pop(-1)
        if num is None:
            return None

        if len(t_range) == 3:
            grid_type, t_start, t_end = t_range
        elif len(t_range) == 2:
            t_start, t_end = t_range
            grid_type = self.default_t_grid_type
            try:
                float(t_start)
            except ValueError:
                grid_type, t_end = t_range
                t_start = self.time_range[0]
        elif len(t_range) == 0:
            t_start, t_end = self.time_range
        else:
            raise ValueError("time range specfication invalid")

        if "lin" in grid_type:
            return np.linspace(float(t_start), float(t_end), int(num))
        elif ("log" in grid_type) or ("geo" in grid_type):
            return np.geomspace(float(t_start), float(t_end), int(num))

    def setup_systematics_sampling(self, priors):
        """Pick how the budget will be evaluated, and check the priors exist.

        Whether the budget is a single sampled value or a set of time nodes
        is settled once here, so that every later evaluation goes straight
        to the right routine.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.

        Raises
        ------
        AssertionError
            If a prior the configuration relies on is missing.
        """
        name, time_range = self.get_name_and_times("", self.systematics_dict)
        if time_range is None:
            self.base_prior_name = name
            assert name in priors, "Required systematics prior missing"
            self.compute_em_err = self.from_param
        else:
            self.err_params = [f"{name}_{i}" for i, _ in enumerate(time_range)]
            assert all(p in priors for p in self.err_params), (
                "Required systematics prior missing"
            )
            self.time_nodes = time_range
            self.compute_em_err = self.from_parameters

    def from_param(self, parameters):
        """Evaluate a budget that is one sampled value, constant in time.

        Parameters
        ----------
        parameters: dict
            Sampled parameters.

        Returns
        -------
        numpy.ndarray
            The budget, one value per observing time.
        """
        return np.full_like(self.light_curve_times, parameters[self.base_prior_name])

    def from_parameters(self, parameters):
        """Evaluate a budget sampled on time nodes, by interpolation.

        Parameters
        ----------
        parameters: dict
            Sampled parameters, one per time node.

        Returns
        -------
        numpy.ndarray
            The budget at each observing time, held constant beyond the
            outermost nodes.
        """
        err_params = [parameters[p] for p in self.err_params]
        return autocomplete_data(
            self.light_curve_times, self.time_nodes, err_params, extrapolate="constant"
        )

    def legacy_prior_setup(self, prior_dict):
        """Read the older ``config`` form of the systematics file.

        Kept so that configuration files written before the current format
        keep working. New files should use the flat structure instead.

        Parameters
        ----------
        prior_dict: bilby.core.prior.PriorDict
            Priors to extend, modified in place.

        Returns
        -------
        bilby.core.prior.PriorDict
            The priors, with the systematics ones added.
        """
        prior_info = get_prior_strings(self.systematics_dict)

        add_prior = dict()
        for line in prior_info:
            line.replace(" ", "")
            elements = line.split("=")
            key = elements[0].replace(" ", "")
            val = "=".join(elements[1:]).strip()
            add_prior[key] = val
        prior_dict.update(bprior.PriorDict(add_prior))
        return prior_dict

    def reset(self, model_times, priors):
        """Rebind the handler to a model whose time span is now known.

        The configuration may describe its time grid relative to the model,
        which is only known once the model exists, hence this second pass.

        Parameters
        ----------
        model_times: numpy.ndarray
            Times the model covers, in days.
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.
        """
        self.time_range = (model_times[0], model_times[-1])
        if self.systematics_dict:
            self.setup_systematics_sampling(priors)
        elif self.base_prior_name in priors:
            self.compute_em_err = self.from_param


class FilterSystematicsHandler(SystematicsHandler):
    """Model the systematics budget of a multi-band light curve.

    Filters do not share the same modelling weaknesses, so each can carry
    its own budget. The configuration allows a single global budget, one per
    filter, one shared by a named group, or one for whatever filters are
    left over.
    """

    def __init__(
        self,
        filters,
        systematics_file=None,
        error_budget=None,
        light_curve_times=np.linspace(0.1, 14, 10),
        base_prior_name="em_syserr",
    ):
        self.filters = filters
        if not isinstance(light_curve_times, dict):
            light_curve_times = {filt: light_curve_times for filt in filters}
        super().__init__(
            systematics_file, error_budget, light_curve_times, base_prior_name
        )

    def adjust_error_budget(self, error_budget):
        """Spread the budget over the times, filter by filter.

        Parameters
        ----------
        error_budget: float or str or dict or None
            One budget for all filters, or one per filter.
        """
        if error_budget is None:
            error_budget = 1.0
        elif isinstance(error_budget, str):
            error_budget = literal_eval(error_budget)
        error_budget = set_filter_associated_dict(error_budget, self.filters, 1.0)
        self.error_budget = {
            filt: np.full_like(self.light_curve_times[filt], error_budget[filt])
            for filt in self.filters
        }

    def setup_systematics_sampling(self, priors):
        """Work out which prior drives which filter, and how.

        Every filter must end up covered, either explicitly, through a
        group, or by the catch-all entry. Once that is settled, the way the
        budget will be evaluated is chosen once and for all: constant,
        one value per filter, interpolated, or a mix of the last two.

        Parameters
        ----------
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.

        Raises
        ------
        AssertionError
            If a filter is left without a budget, or if a prior the
            configuration relies on is missing.
        """
        self.direct_sys_map = {}
        self.interpolate_map = {}
        self.missing_filters = set(self.filters)
        cleared = False

        for key, info_dict in self.systematics_dict.items():
            if key == "config":
                self.legacy_systematics_setup(self.systematics_dict)
                break
            # case 0: only one global systematic uncertainty
            elif key in self.allowed_keys:
                name, time_range = self.get_name_and_times("", self.systematics_dict)
                for filt in self.filters:
                    self.check_names_and_times(filt, time_range, name, priors)
                break

            # case 1: key corresponds to a known filter
            elif key in self.filters:
                name, time_range = self.get_name_and_times(key, info_dict)
                self.check_names_and_times(key, time_range, name, priors)

            # case 2: systematic uncertainties shared by specified filters
            elif "filters" in info_dict:
                name, time_range = self.get_name_and_times(key, info_dict)
                for filt in info_dict["filters"]:
                    self.check_names_and_times(filt, time_range, name, priors)

            # case 3: systematic uncertainties for each filter individually
            elif "each" in info_dict:
                name, time_range = self.get_name_and_times(key, info_dict)
                for filt in info_dict["each"]:
                    filt_name = name.replace(key, filt)
                    self.check_names_and_times(filt, time_range, filt_name, priors)
            # case 4: systematic uncertainties shared by remaining filters
            else:
                cleared = True
                name, time_range = self.get_name_and_times(key, info_dict)
                for filt in self.missing_filters:
                    self.check_names_and_times(
                        filt, time_range, name, priors, clean=False
                    )

        assert cleared or len(self.missing_filters) == 0, (
            f"Some filters are missing systematic uncertainty definitions: {self.missing_filters}"
        )
        if not self.interpolate_map:
            if len(set(self.direct_sys_map.values())) == 1:
                self.compute_em_err = self.from_param
            else:
                self.compute_em_err = self.from_single_params
        elif not self.direct_sys_map:
            self.compute_em_err = self.from_interpolated_params
        else:
            self.compute_em_err = self.from_parameters

    def check_names_and_times(self, filt, time_range, prior_name, priors, clean=True):
        """Assign one filter to its priors, checking they exist.

        Parameters
        ----------
        filt: str
            Filter being assigned.
        time_range: numpy.ndarray or None
            Time nodes, or None for a budget constant in time.
        prior_name: str
            Name of the prior, or its stem when there are time nodes.
        priors: bilby.core.prior.PriorDict
            Priors the analysis will sample.
        clean: bool, optional
            If True, drop any earlier assignment of this filter. The
            catch-all entry passes False, so as not to undo the explicit
            assignments it comes after.

        Raises
        ------
        AssertionError
            If a required prior is missing.
        """
        if clean:
            self.direct_sys_map.pop(filt, None)
            self.interpolate_map.pop(filt, None)
            self.missing_filters.remove(filt)
        if time_range is None:
            assert prior_name in priors, "Required systematics prior missing"
            self.direct_sys_map[filt] = prior_name
        else:
            prior_names = [f"{prior_name}_{i}" for i, _ in enumerate(time_range)]
            for p in prior_names:
                assert p in priors, f"Required systematics prior missing: {p}"
            self.interpolate_map[filt] = (prior_names, time_range)

    def from_param(self, parameters):
        """Evaluate a single budget shared by every filter.

        Parameters
        ----------
        parameters: dict
            Sampled parameters.

        Returns
        -------
        dict
            The budget per filter, constant in time.
        """
        em_err = parameters[self.base_prior_name]
        return {
            filt: np.full_like(self.light_curve_times[filt], em_err)
            for filt in self.filters
        }

    def from_single_params(self, parameters):
        """Evaluate one budget per filter, each constant in time.

        Parameters
        ----------
        parameters: dict
            Sampled parameters.

        Returns
        -------
        dict
            The budget per filter.
        """
        return {
            filt: np.full_like(self.light_curve_times[filt], parameters[prior_name])
            for filt, prior_name in self.direct_sys_map.items()
        }

    def from_interpolated_params(self, parameters):
        """Evaluate budgets sampled on time nodes, by interpolation.

        Parameters
        ----------
        parameters: dict
            Sampled parameters, several per filter.

        Returns
        -------
        dict
            The budget per filter, at each observing time.
        """
        return {
            filt: autocomplete_data(
                self.light_curve_times[filt],
                time_range,
                [parameters[p] for p in param_list],
                extrapolate="constant",
            )
            for filt, (param_list, time_range) in self.interpolate_map.items()
        }

    def from_parameters(self, parameters):
        """Evaluate a mix of constant and time-dependent budgets.

        Parameters
        ----------
        parameters: dict
            Sampled parameters.

        Returns
        -------
        dict
            The budget per filter.
        """
        err_dict = self.from_single_params(parameters)
        err_dict.update(self.from_interpolated_params(parameters))
        return err_dict

    def legacy_systematics_setup(self, systematics_dict):
        """Read the older ``config`` form, for multi-band light curves.

        Kept so that configuration files written before the current format
        keep working.

        Parameters
        ----------
        systematics_dict: dict
            The configuration, in its legacy form.
        """
        validate_only_one_true(systematics_dict)
        time_dep_sys_dict = systematics_dict["config"]["withTime"]
        # case A: no time dependency systematics
        if not time_dep_sys_dict["value"]:
            self.direct_sys_map = {filt: self.base_prior_name for filt in self.filters}
            self.missing_filters = set()
            return

        # case B: time-dependent systematics
        # get the time nodes and the filters
        yaml_filters = list(time_dep_sys_dict["filters"])
        validate_filters(yaml_filters)

        # iterate over the filters and assign them to a systematics filter group
        systematics_filters = {}
        for filter_group in yaml_filters:
            # this should only be the case if no filters are specified
            if filter_group is None:
                systematics_filters = {filt: "all" for filt in self.filters}
                self.missing_filters = set()
                break
            elif isinstance(filter_group, list):
                for filt in filter_group:
                    self.missing_filters.remove(filt)
                    systematics_filters[filt] = "___".join(filter_group)
            else:
                # this should mean that the filter_group is in fact a single filter
                systematics_filters[filter_group] = filter_group
                self.missing_filters.remove(filter_group)

        # By this procedure, every filter should immediately be assigned to a systematics filter-group that we can use to calculate the systematics error
        time_nodes = np.round(
            np.linspace(*self.time_range, time_dep_sys_dict["time_nodes"]), decimals=2
        )
        self.interpolate_map = {
            filt: (
                [
                    f"{self.base_prior_name}_{name}_{i}"
                    for i, _ in enumerate(time_nodes)
                ],
                time_nodes,
            )
            for filt, name in systematics_filters.items()
        }


######### LEGACY  ##################  # noqa: E266

ALLOWED_FILTERS = [
    "2massh",
    "2massj",
    "2massks",
    "atlasc",
    "atlaso",
    "bessellb",
    "besselli",
    "bessellr",
    "bessellux",
    "bessellv",
    "ps1::g",
    "ps1::i",
    "ps1::r",
    "ps1::y",
    "ps1::z",
    "sdssu",
    "uvot::b",
    "uvot::u",
    "uvot::uvm2",
    "uvot::uvw1",
    "uvot::uvw2",
    "uvot::v",
    "uvot::white",
    "ztfg",
    "ztfi",
    "ztfr",
]

ALLOWED_DISTRIBUTIONS = dict(inspect.getmembers(bprior.analytical, inspect.isclass))


def get_positional_args(cls):
    """Names of the arguments a class requires, defaults excluded.

    Used to work out which parameters a bilby prior insists on, so that a
    configuration can be checked before anything is instantiated.

    Parameters
    ----------
    cls: type
        Class to inspect.

    Returns
    -------
    list of str
        Required argument names, self excluded.
    """
    init_method = cls.__init__

    signature = inspect.signature(init_method)
    params = [
        param.name
        for param in signature.parameters.values()
        if param.name != "self" and param.default == inspect.Parameter.empty
    ]

    return params


DISTRIBUTION_PARAMETERS = {
    k: get_positional_args(v) for k, v in ALLOWED_DISTRIBUTIONS.items()
}


def validate_only_one_true(yaml_dict):
    """Check that exactly one legacy configuration mode is enabled.

    Parameters
    ----------
    yaml_dict: dict
        The configuration, in its legacy form.

    Raises
    ------
    ValidationError
        If an entry has no boolean value, or if none or several are enabled.
    """
    for key, values in yaml_dict["config"].items():
        if "value" not in values or not isinstance(values["value"], bool):
            raise ValidationError(key, "'value' key must be present and be a boolean")
    true_count = sum(value["value"] for value in yaml_dict["config"].values())
    if true_count > 1:
        raise ValidationError(
            "config", "Only one configuration key can be set to True at a time"
        )
    elif true_count == 0:
        raise ValidationError(
            "config", "At least one configuration key must be set to True"
        )


def validate_filters(filter_groups):
    """Check that the filter groups are known and do not overlap.

    A filter may belong to one group only: sharing it between two would
    leave it with two budgets and no way to choose.

    Parameters
    ----------
    filter_groups: list
        Groups of filters, each a list or a single filter.

    Raises
    ------
    ValidationError
        If a filter is unknown, or appears in more than one group.
    """
    used_filters = set()
    for filter_group in filter_groups:
        if isinstance(filter_group, list):
            filters_in_group = set()  # Keep track of filters within this group
            for filt in filter_group:
                if filt not in ALLOWED_FILTERS:
                    raise ValidationError(
                        "filters",
                        f"Invalid filter value '{filt}'. Allowed values are {', '.join([str(f) for f in ALLOWED_FILTERS])}",
                    )
                if filt in filters_in_group:
                    raise ValidationError(
                        "filters",
                        f"Duplicate filter value '{filt}' within the same group.",
                    )
                if filt in used_filters:
                    raise ValidationError(
                        "filters",
                        f"Duplicate filter value '{filt}'. A filter can only be used in one group.",
                    )
                used_filters.add(filt)
                filters_in_group.add(filt)
                # Add the filter to the set of used filters within this group
        elif filter_group is not None and filter_group not in ALLOWED_FILTERS:
            raise ValidationError(
                "filters",
                f"Invalid filter value '{filter_group}'. Allowed values are {', '.join([str(f) for f in ALLOWED_FILTERS])}",
            )
        elif filter_group in used_filters:
            raise ValidationError(
                "filters",
                f"Duplicate filter value '{filter_group}'. A filter can only be used in one group.",
            )
        else:
            used_filters.add(filter_group)


def validate_distribution(distribution):
    """Check that a distribution is known and fully specified.

    Parameters
    ----------
    distribution: dict
        Its type, and the parameters that type requires.

    Raises
    ------
    ValidationError
        If the type is unknown, or a required parameter is missing.
    """
    dist_type = distribution.get("type")
    if dist_type not in ALLOWED_DISTRIBUTIONS:
        raise ValidationError(
            "distribution type",
            f"Invalid distribution '{dist_type}'. Allowed values are {', '.join([str(f) for f in ALLOWED_DISTRIBUTIONS])}",
        )

    required_params = DISTRIBUTION_PARAMETERS[dist_type]

    missing_params = set(required_params) - set(distribution.keys())
    if missing_params:
        raise ValidationError(
            "distribution",
            f"Missing required parameters for {dist_type} distribution: {', '.join(missing_params)}",
        )


def create_prior_string(name, distribution):
    """Write one prior as the string a bilby PriorDict reads.

    Parameters the distribution does not use are dropped, with a warning
    rather than an error, so a slightly over-specified file still works.

    Parameters
    ----------
    name: str
        Name to give the prior.
    distribution: dict
        Its type and parameters.

    Returns
    -------
    str
        A line of the form ``name = Uniform(...)``.
    """
    dist_type = distribution["type"]
    prior_class = ALLOWED_DISTRIBUTIONS[dist_type]
    required_params = DISTRIBUTION_PARAMETERS[dist_type]
    params = {
        k: v
        for k, v in distribution.items()
        if k not in ["type", "value", "time_nodes", "filters"]
    }

    extra_params = set(params.keys()) - set(required_params)
    if extra_params:
        warnings.warn(
            f"Distribution parameters {extra_params} are not used by {dist_type} distribution and will be ignored"
        )

    params = {k: params[k] for k in required_params if k in params}

    return f"{name} = {repr(prior_class(**params, name=name))}"


def handle_withTime(values):
    """Build the prior strings of a legacy time-dependent budget.

    One prior is written per filter group and per time node. A group of
    several filters is named by joining them, and a group left unnamed
    becomes ``all``.

    Parameters
    ----------
    values: dict
        The configuration entry: its distribution, its filter groups and
        its number of time nodes.

    Returns
    -------
    list of str
        One prior string per filter group and time node.
    """
    validate_distribution(values)
    filter_groups = values.get("filters", [])
    validate_filters(filter_groups)
    result = []
    time_nodes = values["time_nodes"]

    for filter_group in filter_groups:
        if isinstance(filter_group, list):
            filter_name = "___".join(filter_group)
        else:
            filter_name = filter_group if filter_group is not None else "all"

        for n in range(time_nodes):
            prior_name = f"em_syserr_{filter_name}_{n}"
            result.append(create_prior_string(prior_name, values.copy()))

    return result


def handle_withoutTime(values):
    """Build the single prior string of a legacy constant budget.

    Parameters
    ----------
    values: dict
        The configuration entry, holding its distribution.

    Returns
    -------
    list of str
        A single prior string.
    """
    validate_distribution(values)
    return [create_prior_string("em_syserr", values)]


config_handlers = {
    "withTime": handle_withTime,
    "withoutTime": handle_withoutTime,
}


def get_prior_strings(yaml_dict):
    """Build every prior string a legacy configuration calls for.

    Parameters
    ----------
    yaml_dict: dict
        The configuration, in its legacy form.

    Returns
    -------
    list of str
        Prior strings, ready for a bilby PriorDict.
    """
    validate_only_one_true(yaml_dict)
    results = []
    for key, values in yaml_dict["config"].items():
        if values["value"] and key in config_handlers:
            results.extend(config_handlers[key](values))
    return results


def main(yaml_file_path):
    """Read a legacy systematics file and return its prior strings.

    Parameters
    ----------
    yaml_file_path: str
        Path to the configuration file.

    Returns
    -------
    list of str
        Prior strings, ready for a bilby PriorDict.
    """
    return get_prior_strings(load_yaml(yaml_file_path))
