import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import torch

    from nmma.mlmodel import dataprocessing

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is the optional neuralnet extra
    TORCH_AVAILABLE = False

needs_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch is not installed; install the neuralnet extra"
)


def lightcurve_json(num_points=20, start=0.0, step=0.25, magnitude=20.0):
    """The shape nmma's light curve generation writes: one entry per band,
    each a list of [time, magnitude, uncertainty] triples."""
    times = [start + index * step for index in range(num_points)]
    return {
        band: [[time, magnitude, 0.1] for time in times]
        for band in ["ztfg", "ztfr", "ztfi"]
    }


@needs_torch
class TestModuleConstants:
    """The pipeline is pinned to three ZTF bands on a fixed time grid,
    because the released weights were trained that way."""

    def test_three_bands_give_three_channels(self):
        assert dataprocessing.bands == ["ztfg", "ztfr", "ztfi"]
        assert dataprocessing.num_channels == 3

    def test_the_grid_length_follows_the_span_and_the_step(self):
        # Every light curve is padded to this many points, so the number has
        # to agree with the time span the padding walks.
        assert dataprocessing.num_points == 121
        assert dataprocessing.days == 30
        assert dataprocessing.time_step == 0.25

    def test_the_detection_limit_doubles_as_the_padding_value(self):
        assert dataprocessing.detection_limit == 22.0

    def test_the_reference_times_are_ordered(self):
        assert dataprocessing.t_min < dataprocessing.t_zero
        assert dataprocessing.t_zero < dataprocessing.t_max


@needs_torch
class TestOpenJson:
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_the_contents_are_returned_as_a_dictionary(self):
        (self.tmp_dir / "lc.json").write_text(json.dumps({"ztfg": [[0.0, 20.0]]}))
        data = dataprocessing.open_json("lc.json", str(self.tmp_dir) + "/")
        assert data == {"ztfg": [[0.0, 20.0]]}

    def test_the_directory_and_the_name_are_joined_by_plain_concatenation(self):
        # There is no separator handling, so the directory argument has to
        # end in one.
        (self.tmp_dir / "lc.json").write_text("{}")
        with pytest.raises(FileNotFoundError):
            dataprocessing.open_json("lc.json", str(self.tmp_dir))

    def test_a_missing_file_is_reported(self):
        with pytest.raises(FileNotFoundError):
            dataprocessing.open_json("absent.json", str(self.tmp_dir) + "/")


@needs_torch
class TestFileNameBuilders:
    """Two naming schemes exist for the generated light curve batches, one
    for the training sets and one for the test sets."""

    def test_the_training_names_nest_a_batch_directory(self):
        names = dataprocessing.get_names("/data", "kn", 2, 3)
        assert names[0] == "/data/kn_batch_2/kn_2_0.json"
        assert names[2] == "/data/kn_batch_2/kn_2_2.json"

    def test_one_name_is_built_per_file(self):
        assert len(dataprocessing.get_names("/data", "kn", 0, 5)) == 5

    def test_asking_for_no_files_gives_no_names(self):
        assert dataprocessing.get_names("/data", "kn", 0, 0) == []

    def test_the_test_names_are_flat(self):
        names = dataprocessing.get_test_names("/data", "kn", 1, 2)
        assert names[0] == "/data/kn1_0.json"
        assert names[1] == "/data/kn1_1.json"

    def test_the_two_schemes_differ(self):
        assert dataprocessing.get_names("/d", "kn", 1, 1) != dataprocessing.get_test_names(
            "/d", "kn", 1, 1
        )


@needs_torch
class TestJsonToDataframe:
    """Each generated light curve json is flattened into one table with a
    column per band, plus a detection count and a simulation number."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.paths = []
        for index in range(2):
            path = self.tmp_dir / f"lc_{index}.json"
            path.write_text(json.dumps(lightcurve_json()))
            self.paths.append(str(path))

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_one_table_is_returned_per_file(self):
        frames = dataprocessing.json_to_df(self.paths, 2)
        assert len(frames) == 2

    def test_each_band_becomes_a_column(self):
        frame = dataprocessing.json_to_df(self.paths, 1)[0]
        for band in ["ztfg", "ztfr", "ztfi"]:
            assert band in frame.columns, band

    def test_the_time_column_is_kept(self):
        frame = dataprocessing.json_to_df(self.paths, 1)[0]
        assert "t" in frame.columns
        assert len(frame) == 20

    def test_the_uncertainty_column_is_dropped(self):
        frame = dataprocessing.json_to_df(self.paths, 1)[0]
        assert "x" not in frame.columns

    def test_every_point_below_the_limit_counts_as_a_detection(self):
        frame = dataprocessing.json_to_df(self.paths, 1)[0]
        assert frame["num_detections"].iloc[0] == 3 * 20

    def test_points_at_the_limit_are_not_detections(self):
        path = self.tmp_dir / "faint.json"
        path.write_text(json.dumps(lightcurve_json(magnitude=22.0)))
        frame = dataprocessing.json_to_df([str(path)], 1)[0]
        assert frame["num_detections"].iloc[0] == 0

    def test_the_detection_count_is_summed_over_every_band(self):
        data = lightcurve_json(num_points=4, magnitude=22.0)
        data["ztfg"] = [[0.0, 19.0, 0.1]] * 4
        path = self.tmp_dir / "mixed.json"
        path.write_text(json.dumps(data))
        frame = dataprocessing.json_to_df([str(path)], 1)[0]
        assert frame["num_detections"].iloc[0] == 4

    def test_the_detection_count_is_repeated_on_every_row(self):
        frame = dataprocessing.json_to_df(self.paths, 1)[0]
        assert frame["num_detections"].nunique() == 1

    def test_each_file_gets_its_own_simulation_number(self):
        frames = dataprocessing.json_to_df(self.paths, 2)
        assert frames[0]["sim_id"].iloc[0] == 0
        assert frames[1]["sim_id"].iloc[0] == 1

    def test_a_custom_detection_limit_is_honoured(self):
        path = self.tmp_dir / "custom.json"
        path.write_text(json.dumps(lightcurve_json(magnitude=21.0)))
        frame = dataprocessing.json_to_df([str(path)], 1, detection_limit=21.0)[0]
        assert frame["num_detections"].iloc[0] == 0


@needs_torch
class TestFillerGeneration:
    """Padding rows carry the detection limit in every band, so a light
    curve that does not span the grid reads as a non-detection there."""

    columns = ["t", "ztfg", "ztfr"]

    def test_the_leading_filler_spans_the_requested_window(self):
        filler = dataprocessing.gen_prepend_filler(self.columns, 22.0, 0.0, 1.0, 0.25)
        np.testing.assert_allclose(filler["t"], [0.0, 0.25, 0.5, 0.75])

    def test_the_leading_filler_stops_before_the_end_of_the_window(self):
        filler = dataprocessing.gen_prepend_filler(self.columns, 22.0, 0.0, 1.0, 0.25)
        assert filler["t"].max() < 1.0

    def test_every_band_is_filled_with_the_limit(self):
        filler = dataprocessing.gen_prepend_filler(self.columns, 22.0, 0.0, 1.0, 0.25)
        for band in ["ztfg", "ztfr"]:
            assert (filler[band] == 22.0).all(), band

    def test_the_column_order_is_preserved(self):
        filler = dataprocessing.gen_prepend_filler(self.columns, 22.0, 0.0, 1.0)
        assert filler.columns.tolist() == self.columns

    def test_an_empty_window_gives_no_filler(self):
        filler = dataprocessing.gen_prepend_filler(self.columns, 22.0, 1.0, 1.0, 0.25)
        assert len(filler) == 0

    def test_the_trailing_filler_is_built_from_a_count_not_a_window(self):
        # The trailing pad has to land on exactly the number of rows still
        # missing from the grid.
        filler = dataprocessing.gen_append_filler(self.columns, 22.0, 5.0, 4, 0.25)
        assert len(filler) == 4
        np.testing.assert_allclose(filler["t"], [5.0, 5.25, 5.5, 5.75])

    def test_the_trailing_filler_also_carries_the_limit(self):
        filler = dataprocessing.gen_append_filler(self.columns, 22.0, 5.0, 3)
        assert (filler["ztfg"] == 22.0).all()

    def test_asking_for_no_trailing_rows_gives_none(self):
        filler = dataprocessing.gen_append_filler(self.columns, 22.0, 5.0, 0)
        assert len(filler) == 0


@needs_torch
class TestPadTheData:
    """The network needs every light curve on the same fixed-length grid, so
    observations are shifted to start at zero and padded on both ends."""

    columns = ["t", "ztfg", "ztfr", "ztfi"]

    def frame(self, num_points=20, start=0.0, step=0.25):
        times = np.arange(num_points) * step + start + dataprocessing.t_min
        return pd.DataFrame(
            {
                "t": times,
                "ztfg": np.full(num_points, 20.0),
                "ztfr": np.full(num_points, 20.0),
                "ztfi": np.full(num_points, 20.0),
            }
        )

    def test_the_grid_length_is_reached(self):
        padded = dataprocessing.pad_the_data(self.frame(), self.columns)
        assert len(padded) == dataprocessing.num_points

    def test_the_times_are_shifted_to_start_from_the_reference_time(self):
        padded = dataprocessing.pad_the_data(self.frame(), self.columns)
        assert padded["t"].min() == pytest.approx(0.0)

    def test_a_shorter_grid_can_be_requested(self):
        padded = dataprocessing.pad_the_data(
            self.frame(), self.columns, desired_count=40
        )
        assert len(padded) == 40

    def test_the_observations_are_kept(self):
        padded = dataprocessing.pad_the_data(self.frame(), self.columns)
        assert int((padded["ztfg"] == 20.0).sum()) == 20

    def test_the_padding_is_the_detection_limit(self):
        padded = dataprocessing.pad_the_data(self.frame(), self.columns)
        assert int((padded["ztfg"] == dataprocessing.detection_limit).sum()) == (
            dataprocessing.num_points - 20
        )

    def test_a_late_first_observation_is_padded_at_the_front(self):
        padded = dataprocessing.pad_the_data(self.frame(start=5.0), self.columns)
        assert padded["t"].min() == pytest.approx(0.0)
        assert padded["ztfg"].iloc[0] == dataprocessing.detection_limit

    def test_an_observation_starting_at_the_reference_time_is_not_front_padded(self):
        padded = dataprocessing.pad_the_data(self.frame(start=0.0), self.columns)
        assert padded["ztfg"].iloc[0] == 20.0

    def test_the_trailing_padding_continues_the_time_grid_evenly(self):
        padded = dataprocessing.pad_the_data(self.frame(start=0.0), self.columns)
        spacing = np.diff(padded["t"].to_numpy())
        np.testing.assert_allclose(spacing, 0.25, atol=1e-6)

    def test_the_leading_padding_leaves_a_gap_before_the_first_observation(self):
        # The filler is built with numpy.arange up to one step before the
        # first observation, but arange already excludes its stop value, so
        # one row is lost and the join is a double step. Stopping at the
        # first observation time instead would close it. Only the time
        # column is affected, since the network is fed the magnitudes.
        padded = dataprocessing.pad_the_data(self.frame(start=5.0), self.columns)
        spacing = np.diff(padded["t"].to_numpy())
        oversized = np.flatnonzero(np.abs(spacing - 0.25) > 1e-9)
        assert len(oversized) == 1
        assert spacing[oversized[0]] == pytest.approx(0.5)

    def test_a_custom_filler_value_is_used(self):
        padded = dataprocessing.pad_the_data(
            self.frame(), self.columns, filler_data=99.0
        )
        assert 99.0 in padded["ztfg"].tolist()

    def test_a_light_curve_already_longer_than_the_grid_is_refused(self):
        # The function asserts the final length, so it cannot trim.
        with pytest.raises(AssertionError):
            dataprocessing.pad_the_data(
                self.frame(num_points=200), self.columns, desired_count=121
            )


@needs_torch
class TestPadAllDataframes:
    def test_padding_a_list_of_light_curves_fails(self):
        # pad_all_dfs calls pad_the_data(df) but that function requires a
        # column list as its second argument, so the helper raises for any
        # input. Forwarding df.columns.to_list() would fix it.
        frame = pd.DataFrame(
            {
                "t": np.arange(20) * 0.25 + dataprocessing.t_min,
                "ztfg": np.full(20, 20.0),
                "num_detections": np.full(20, 3),
                "sim_id": np.zeros(20, dtype=int),
            }
        )
        with pytest.raises(TypeError):
            dataprocessing.pad_all_dfs([frame])

    def test_an_empty_list_is_handled_because_nothing_is_padded(self):
        assert dataprocessing.pad_all_dfs([]) == []


@needs_torch
class CsvFixtureMixin:
    """The later stages read the padded light curves back from csv files,
    one simulation per block of rows."""

    num_points = 4
    num_repeats = 2

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.data_dir = str(self.tmp_dir) + "/"

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def write_csv(self, name, index, num_sims=4, offset=0.0):
        rows = num_sims * self.num_points
        frame = pd.DataFrame(
            {
                "t": np.tile(np.arange(self.num_points) * 0.25, num_sims),
                "ztfg": np.full(rows, 20.0 + offset),
                "ztfr": np.full(rows, 21.0 + offset),
                "ztfi": np.full(rows, 22.0 + offset),
                "num_detections": np.full(rows, 3),
                "log10_mej": np.full(rows, -2.0 + offset),
                "log10_vej": np.full(rows, -1.0 + offset),
                "log10_Xlan": np.full(rows, -3.0 + offset),
                "extra_a": np.full(rows, 1.0),
                "extra_b": np.full(rows, 2.0),
                "extra_c": np.full(rows, 3.0),
                "sim_id": np.repeat(np.arange(num_sims), self.num_points),
            }
        )
        frame.to_csv(self.tmp_dir / f"{name}_{index}.csv", index=False)
        return frame


@needs_torch
class TestLoadInData(CsvFixtureMixin):
    def test_several_files_are_concatenated(self):
        self.write_csv("lc", 0)
        self.write_csv("lc", 1)
        frame = dataprocessing.load_in_data(
            self.data_dir, "lc", 2, self.num_points, self.num_repeats
        )
        assert len(frame) == 2 * 4 * self.num_points

    def test_every_light_curve_is_renumbered_across_the_files(self):
        self.write_csv("lc", 0)
        self.write_csv("lc", 1)
        frame = dataprocessing.load_in_data(
            self.data_dir, "lc", 2, self.num_points, self.num_repeats
        )
        assert frame["sim_id"].nunique() == 8
        assert frame["sim_id"].tolist()[: self.num_points] == [0] * 4

    def test_light_curves_are_grouped_into_batches_of_repeats(self):
        self.write_csv("lc", 0)
        frame = dataprocessing.load_in_data(
            self.data_dir, "lc", 1, self.num_points, self.num_repeats
        )
        assert frame["batch_id"].nunique() == 2

    def test_each_batch_holds_one_set_of_repeats(self):
        self.write_csv("lc", 0)
        frame = dataprocessing.load_in_data(
            self.data_dir, "lc", 1, self.num_points, self.num_repeats
        )
        counts = frame.groupby("batch_id").size().unique().tolist()
        assert counts == [self.num_points * self.num_repeats]

    def test_light_curves_beyond_a_whole_batch_are_dropped(self):
        # A partial batch cannot be reshaped into the repeat axis, so the
        # tail is cut rather than padded.
        self.write_csv("lc", 0, num_sims=5)
        frame = dataprocessing.load_in_data(
            self.data_dir, "lc", 1, self.num_points, self.num_repeats
        )
        assert len(frame) == 4 * self.num_points


@needs_torch
class TestMatchFixToVar(CsvFixtureMixin):
    """The contrastive training needs each time-shifted light curve paired
    with its unshifted counterpart, matched on the simulation number."""

    def match(self):
        self.write_csv("var", 0, offset=0.0)
        self.write_csv("fix", 0, offset=0.5)
        return dataprocessing.match_fix_to_var(
            self.data_dir, "var", "fix", 0, 1, self.num_points, self.num_repeats
        )

    def test_both_halves_are_returned(self):
        fixed, varied = self.match()
        assert len(fixed) == len(varied)

    def test_the_pairs_are_aligned_row_by_row(self):
        fixed, varied = self.match()
        assert fixed["sim_id"].tolist() == varied["sim_id"].tolist()

    def test_both_halves_are_numbered_and_batched_identically(self):
        fixed, varied = self.match()
        assert fixed["batch_id"].tolist() == varied["batch_id"].tolist()

    def test_the_merge_suffixes_are_stripped_from_the_column_names(self):
        fixed, varied = self.match()
        assert "sim_id" in fixed.columns
        assert "sim_id" in varied.columns

    def test_the_suffix_stripping_also_eats_trailing_band_letters(self):
        # The columns are cleaned with rstrip("_y"), which removes any
        # trailing underscore or y rather than the two-character suffix, so
        # a column whose own name ends in y loses that letter too.
        assert (
            pd.Index(["ztfy_y", "key_y", "ztfg_y"]).str.rstrip("_y").tolist()
            == ["ztf", "ke", "ztfg"]
        )

    def test_a_partial_batch_is_dropped_from_both_halves(self):
        self.write_csv("var", 0, num_sims=5)
        self.write_csv("fix", 0, num_sims=5, offset=0.5)
        fixed, varied = dataprocessing.match_fix_to_var(
            self.data_dir, "var", "fix", 0, 1, self.num_points, self.num_repeats
        )
        assert len(fixed) == 4 * self.num_points
        assert len(varied) == 4 * self.num_points


@needs_torch
class TestMatched(CsvFixtureMixin):
    def test_the_two_halves_stay_in_one_table(self):
        self.write_csv("var", 0)
        self.write_csv("fix", 0, offset=0.5)
        frame = dataprocessing.matched(self.data_dir, "var", "fix", 0, 1)
        assert len(frame) == 4 * self.num_points
        assert any(column.endswith("_x") for column in frame.columns)
        assert any(column.endswith("_y") for column in frame.columns)

    def test_several_files_are_concatenated(self):
        for index in range(2):
            self.write_csv("var", index)
            self.write_csv("fix", index, offset=0.5)
        frame = dataprocessing.matched(self.data_dir, "var", "fix", 0, 2)
        assert len(frame) == 2 * 4 * self.num_points


@needs_torch
class TestAddBatchSimNumbers:
    """Numbering is rewritten in place after a table has been split or
    recombined, so the batch axis still lines up with the repeats."""

    def frame(self, num_sims=4, num_points=4):
        return pd.DataFrame({"ztfg": np.zeros(num_sims * num_points)})

    def test_the_numbers_are_added_in_place(self):
        frame = self.frame()
        assert (
            dataprocessing.add_batch_sim_nums_all(frame, num_points=4, num_repeats=2)
            is None
        )
        assert "sim_id" in frame.columns
        assert "batch_id" in frame.columns

    def test_one_simulation_number_is_given_per_block_of_points(self):
        frame = self.frame()
        dataprocessing.add_batch_sim_nums_all(frame, num_points=4, num_repeats=2)
        assert frame["sim_id"].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4

    def test_one_batch_number_is_given_per_set_of_repeats(self):
        frame = self.frame()
        dataprocessing.add_batch_sim_nums_all(frame, num_points=4, num_repeats=2)
        assert frame["batch_id"].tolist() == [0] * 8 + [1] * 8

    def test_a_table_that_is_not_a_whole_number_of_batches_is_refused(self):
        frame = self.frame(num_sims=3)
        with pytest.raises(ValueError):
            dataprocessing.add_batch_sim_nums_all(frame, num_points=4, num_repeats=2)


@needs_torch
class TestRepeatedDataframeToTensor:
    """The final step before training: each batch becomes a tensor with the
    repeat axis first and the bands as channels."""

    num_points = 4
    num_repeats = 2
    num_batches = 2

    def setup_method(self):
        self.original_points = dataprocessing.num_points
        self.original_repeats = dataprocessing.num_repeats
        dataprocessing.num_points = self.num_points
        dataprocessing.num_repeats = self.num_repeats

    def teardown_method(self):
        dataprocessing.num_points = self.original_points
        dataprocessing.num_repeats = self.original_repeats

    def frame(self, offset=0.0):
        rows = self.num_batches * self.num_repeats * self.num_points
        data = {
            "t": np.tile(np.arange(self.num_points) * 0.25, rows // self.num_points)
        }
        for index, band in enumerate(["ztfg", "ztfr", "ztfi"]):
            data[band] = np.full(rows, 20.0 + index + offset)
        for index, name in enumerate(
            ["pad_a", "par_0", "par_1", "par_2", "par_3", "par_4", "pad_b"]
        ):
            data[name] = np.full(rows, float(index) + offset)
        data["batch_id"] = np.repeat(
            np.arange(self.num_batches), self.num_repeats * self.num_points
        )
        return pd.DataFrame(data)

    def convert(self):
        return dataprocessing.repeated_df_to_tensor(
            self.frame(), self.frame(offset=0.5), self.num_batches
        )

    def test_four_lists_are_returned(self):
        assert len(self.convert()) == 4

    def test_one_tensor_is_produced_per_batch(self):
        for entry in self.convert():
            assert len(entry) == self.num_batches

    def test_the_light_curves_carry_the_bands_as_channels(self):
        shifted, unshifted, _, _ = self.convert()
        for tensor in [shifted[0], unshifted[0]]:
            assert tuple(tensor.shape) == (self.num_repeats, 3, self.num_points)

    def test_the_parameters_carry_one_row_per_repeat(self):
        _, _, shifted_parameters, unshifted_parameters = self.convert()
        for tensor in [shifted_parameters[0], unshifted_parameters[0]]:
            assert tuple(tensor.shape) == (self.num_repeats, 1, 5)

    def test_the_tensors_are_single_precision_floats(self):
        for entry in self.convert():
            assert entry[0].dtype == torch.float32

    def test_the_band_values_survive_the_reshape(self):
        shifted, _, _, _ = self.convert()
        for channel, expected in enumerate([20.0, 21.0, 22.0]):
            torch.testing.assert_close(
                shifted[0][:, channel, :],
                torch.full((self.num_repeats, self.num_points), expected),
            )

    def test_the_shifted_and_unshifted_parameters_are_read_from_different_columns(self):
        # The shifted table carries one extra leading column, so the
        # parameter block sits one position further along.
        _, _, shifted_parameters, unshifted_parameters = self.convert()
        assert not torch.equal(shifted_parameters[0], unshifted_parameters[0])


@needs_torch
class TestPaperDataset:
    """The dataset the training loops iterate over. It yields parameters and
    light curves for both the shifted and unshifted halves."""

    def build(self, num_batches=3):
        self.light_curves = [torch.randn(2, 3, 4) for _ in range(num_batches)]
        self.parameters = [torch.randn(2, 1, 5) for _ in range(num_batches)]
        return dataprocessing.Paper_data(
            self.light_curves,
            [tensor.clone() for tensor in self.light_curves],
            self.parameters,
            [tensor.clone() for tensor in self.parameters],
            num_batches,
        )

    def test_the_length_is_the_number_of_batches(self):
        assert len(self.build(num_batches=3)) == 3

    def test_an_item_is_a_four_tuple(self):
        assert len(self.build()[0]) == 4

    def test_the_parameters_come_before_the_light_curves(self):
        # The training loops unpack in this order, so it is part of the
        # contract rather than an implementation detail.
        dataset = self.build()
        shifted_parameters, _, shifted_data, _ = dataset[0]
        torch.testing.assert_close(shifted_parameters, self.parameters[0])
        torch.testing.assert_close(shifted_data, self.light_curves[0])

    def test_a_tensor_index_is_accepted(self):
        dataset = self.build()
        torch.testing.assert_close(dataset[torch.tensor(1)][0], self.parameters[1])

    def test_every_batch_can_be_reached(self):
        dataset = self.build(num_batches=3)
        for index in range(len(dataset)):
            assert len(dataset[index]) == 4

    def test_it_works_as_a_torch_dataset(self):
        from torch.utils.data import DataLoader

        loader = DataLoader(self.build(num_batches=2), batch_size=1)
        batches = list(loader)
        assert len(batches) == 2
        assert tuple(batches[0][2].shape) == (1, 2, 3, 4)
