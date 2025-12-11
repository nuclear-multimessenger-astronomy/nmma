from pathlib import Path
import pytest

import yaml
from ..em.systematics import (
    ValidationError,
    validate_only_one_true,
    validate_filters,
    handle_withTime,
    handle_withoutTime,
    main,
    ALLOWED_FILTERS,
    ALLOWED_DISTRIBUTIONS,
)


@pytest.fixture
def sample_yaml_content():
    return """
config:
  withTime:
    value: true
    type: Uniform
    minimum: 0.0
    maximum: 1.0
    time_nodes: 2
    filters:
      - [bessellb, bessellv]
      - ztfr
  withoutTime:
    value: false
    type: Uniform
    minimum: 0.0
    maximum: 1.0
"""


@pytest.fixture
def sample_yaml_file(tmp_path, sample_yaml_content):
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(sample_yaml_content)
    return yaml_file


def test_validate_only_one_true_valid(sample_yaml_file):
    yaml_dict = yaml.safe_load(Path(sample_yaml_file).read_text())
    validate_only_one_true(yaml_dict)  # Should not raise an exception


def test_validate_only_one_true_invalid():
    invalid_yaml = {
        "config": {"withTime": {"value": True}, "withoutTime": {"value": True}}
    }
    with pytest.raises(
        ValidationError, match="Only one configuration key can be set to True at a time"
    ):
        validate_only_one_true(invalid_yaml)


def test_validate_filters_valid():
    valid_filters = [["bessellb", "bessellv"], "ztfr"]
    validate_filters(valid_filters)  # Should not raise an exception


def test_validate_filters_invalid():
    invalid_filters = [["bessellb", "invalid_filter"], "ztfr"]
    with pytest.raises(ValidationError, match="Invalid filter value 'invalid_filter'"):
        validate_filters(invalid_filters)


def test_handle_withTime():
    values = {
        "type": "Uniform",
        "minimum": 0.0,
        "maximum": 1.0,
        "time_nodes": 2,
        "filters": [["bessellb", "bessellv"], "ztfr"],
    }
    result = handle_withTime(values)
    assert "sys_err_bessellb___bessellv1" in result[0]
    assert "sys_err_ztfr2" in result[3]


def test_handle_withoutTime():
    values = {"type": "Uniform", "minimum": 0.0, "maximum": 1.0}
    result = handle_withoutTime(values)
    assert len(result) == 1
    assert (
        "sys_err = Uniform(minimum=0.0, maximum=1.0, name='sys_err', latex_label='sys_err', unit=None, boundary=None)"
        in result[0]
    )


def test_main(sample_yaml_file):
    result = main(sample_yaml_file)
    assert all("sys_err" in line for line in result)


def test_main_invalid_yaml(tmp_path):
    invalid_yaml = tmp_path / "invalid_config.yaml"
    invalid_yaml.write_text("invalid: yaml: content")
    with pytest.raises(yaml.YAMLError):
        main(invalid_yaml)


def test_validate_only_one_true_all_false():
    invalid_yaml = {
        "config": {"withTime": {"value": False}, "withoutTime": {"value": False}}
    }
    with pytest.raises(
        ValidationError, match="At least one configuration key must be set to True"
    ):
        validate_only_one_true(invalid_yaml)


def test_validate_only_one_true_missing_value():
    invalid_yaml = {"config": {"withTime": {}, "withoutTime": {"value": False}}}
    with pytest.raises(
        ValidationError, match="'value' key must be present and be a boolean"
    ):
        validate_only_one_true(invalid_yaml)


def test_validate_filters_duplicate_in_group():
    invalid_filters = [["bessellb", "bessellb"], "ztfr"]
    with pytest.raises(
        ValidationError, match="Duplicate filter value 'bessellb' within the same group"
    ):
        validate_filters(invalid_filters)


def test_validate_filters_duplicate_across_groups():
    invalid_filters = [["bessellb", "bessellv"], "bessellb"]
    with pytest.raises(
        ValidationError,
        match="Duplicate filter value 'bessellb'. A filter can only be used in one group",
    ):
        validate_filters(invalid_filters)


def test_validate_filters_none_value():
    valid_filters = [["bessellb", "bessellv"], None]
    validate_filters(valid_filters)  # Should not raise an exception


def test_validate_filters_empty_list():
    validate_filters([])  # Should not raise an exception


def test_validate_distribution_valid():
    assert ALLOWED_DISTRIBUTIONS["Uniform"]  # Should not raise an exception


def test_validate_distribution_invalid():
    with pytest.raises(KeyError):
        assert ALLOWED_DISTRIBUTIONS["nonuniform"]  # Should be "Uniform"


def test_validate_distribution_case_sensitive():
    with pytest.raises(KeyError):
        assert ALLOWED_DISTRIBUTIONS["uniform"]  # Should be "Uniform"


def test_handle_withTime_single_filter():
    values = {
        "type": "Uniform",
        "minimum": 0.0,
        "maximum": 1.0,
        "time_nodes": 2,
        "filters": ["ztfr"],
    }
    result = handle_withTime(values)
    assert len(result) == 2
    assert all("sys_err_ztfr" in line for line in result)


def test_handle_withTime_all_filters():
    values = {
        "type": "Uniform",
        "minimum": 0.0,
        "maximum": 1.0,
        "time_nodes": 1,
        "filters": [None],
    }
    result = handle_withTime(values)
    assert len(result) == 1
    assert "sys_err_all1" in result[0]


def test_main_withoutTime(tmp_path):
    yaml_content = """
config:
  withTime:
    value: false
  withoutTime:
    value: true
    type: Uniform
    minimum: 0.0
    maximum: 1.0
"""
    yaml_file = tmp_path / "withoutTime_config.yaml"
    yaml_file.write_text(yaml_content)
    result = main(yaml_file)
    assert len(result) == 1
    assert "sys_err = Uniform" in result[0]


def test_main_empty_config(tmp_path):
    yaml_content = """
config:
  withTime:
    value: false
  withoutTime:
    value: false
"""
    yaml_file = tmp_path / "empty_config.yaml"
    yaml_file.write_text(yaml_content)
    with pytest.raises(
        ValidationError, match="At least one configuration key must be set to True"
    ):
        main(yaml_file)


@pytest.mark.parametrize("filter_name", ALLOWED_FILTERS)
def test_all_allowed_filters(filter_name):
    values = {
        "type": "Uniform",
        "minimum": 0.0,
        "maximum": 1.0,
        "time_nodes": 1,
        "filters": [filter_name],
    }
    result = handle_withTime(values)
    assert len(result) == 1
    assert f"sys_err_{filter_name}1" in result[0]


def test_load_yaml_file_not_found():
    with pytest.raises(FileNotFoundError):
        main("non_existent_file.yaml")


def test_load_yaml_invalid_format(tmp_path):
    invalid_yaml = tmp_path / "invalid_format.yaml"
    invalid_yaml.write_text("{ invalid: yaml: content")
    with pytest.raises(yaml.YAMLError):
        main(invalid_yaml)
