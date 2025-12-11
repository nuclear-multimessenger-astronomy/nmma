import inspect
import warnings
from pathlib import Path

import yaml
from bilby.core.prior import analytical

warnings.simplefilter("module", DeprecationWarning)


class ValidationError(ValueError):
    def __init__(self, key, message):
        super().__init__(f"Validation error for '{key}': {message}")


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
    "ps1__g",
    "ps1__i",
    "ps1__r",
    "ps1__y",
    "ps1__z",
    "sdssu",
    "uvot__b",
    "uvot__u",
    "uvot__uvm2",
    "uvot__uvw1",
    "uvot__uvw2",
    "uvot__v",
    "uvot__white",
    "ztfg",
    "ztfi",
    "ztfr",
]

ALLOWED_DISTRIBUTIONS = dict(inspect.getmembers(analytical, inspect.isclass))


def get_positional_args(cls):
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


def load_yaml(file_path):
    return yaml.safe_load(Path(file_path).read_text())


def validate_only_one_true(yaml_dict):
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
    dist_type = distribution.pop("type")
    _ = distribution.pop("value", None)
    _ = distribution.pop("time_nodes", None)
    _ = distribution.pop("filters", None)
    prior_class = ALLOWED_DISTRIBUTIONS[dist_type]
    required_params = DISTRIBUTION_PARAMETERS[dist_type]
    params = distribution.copy()

    extra_params = set(params.keys()) - set(required_params)
    if extra_params:
        warnings.warn(
            f"Distribution parameters {extra_params} are not used by {dist_type} distribution and will be ignored"
        )

    params = {k: params[k] for k in required_params if k in params}

    return f"{name} = {repr(prior_class(**params, name=name))}"


def handle_withTime(values):
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

        for n in range(1, time_nodes + 1):
            prior_name = f"sys_err_{filter_name}{n}"
            result.append(create_prior_string(prior_name, values.copy()))

    return result


def handle_withoutTime(values):
    validate_distribution(values)
    return [create_prior_string("sys_err", values)]


config_handlers = {
    "withTime": handle_withTime,
    "withoutTime": handle_withoutTime,
}


def main(yaml_file_path):
    yaml_dict = load_yaml(yaml_file_path)
    validate_only_one_true(yaml_dict)
    results = []
    for key, values in yaml_dict["config"].items():
        if values["value"] and key in config_handlers:
            results.extend(config_handlers[key](values))
    return results
