import yaml
from pathlib import Path


class ValidationError(ValueError):
    def __init__(self, key, message):
        super().__init__(f"Validation error for '{key}': {message}")


ALLOWED_FILTERS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
    "J",
    "H",
    "K",
]  # only optical and IR right now, case sensitive


def load_yaml(file_path):
    return yaml.safe_load(Path(file_path).read_text())


def validate_only_one_true(yaml_dict):
    for key, values in yaml_dict["config"].items():
        if "value" not in values or type(values["value"]) is not bool:
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


def validate_filters(filters):
    for filter_val in filters:
        if filter_val is None:
            continue
        if filter_val not in ALLOWED_FILTERS:
            raise ValidationError(
                "filters",
                f"Invalid filter value '{filter_val}'. Allowed values are {', '.join([str(f) for f in ALLOWED_FILTERS])}",
            )


def validate_distribution(distribution):
    if distribution != "Uniform":
        raise ValidationError(
            "type",
            f"Invalid distribution '{distribution}'. Only 'Uniform' distribution is supported",
        )


def validate_fields(key, values, required_fields):
    missing_fields = [field for field in required_fields if values.get(field) is None]
    if missing_fields:
        raise ValidationError(key, f"Missing fields: {', '.join(missing_fields)}")
    for field, expected_type in required_fields.items():
        if not isinstance(values[field], expected_type):
            raise ValidationError(key, f"'{field}' must be of type {expected_type}")


def handle_withTime(key, values):
    required_fields = {
        "type": str,
        "min": (float, int),
        "max": (float, int),
        "time_nodes": int,
        "filters": list,
    }
    validate_fields(key, values, required_fields)
    filters = values.get("filters", [])
    validate_filters(filters)
    distribution = values.get("type")
    validate_distribution(distribution)
    result = []
    for filter_val in filters:
        filter_str = "all" if filter_val is None else filter_val
        for n in range(1, values["time_nodes"] + 1):
            result.append(
                f'sys_err_{filter_str}{n} = {values["type"]}(minimum={values["min"]},maximum={values["max"]},name="sys_err_{filter_str}{n}")'
            )
    return result


def handle_withoutTime(key, values):
    required_fields = {"type": str, "min": (float, int), "max": (float, int)}
    validate_fields(key, values, required_fields)
    distribution = values.get("type")
    validate_distribution(distribution)
    return [
        f'sys_err = {values["type"]}(minimum={values["min"]},maximum={values["max"]},name="sys_err")'
    ]


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
            results.extend(config_handlers[key](key, values))
    return results
