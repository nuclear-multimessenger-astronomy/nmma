import numpy as np
import pytest

from nmma.eos.tov_emulators import EmulatorFile, TOVEmulator


def random_emulator_file(
    rng: np.random.Generator,
    eos_parameters: list[str],
    units: int,
    n_points: int,
    n_components: int,
) -> EmulatorFile:
    """A small random emulator with two hidden layers of ``units`` units,
    ``2 * n_components + 1`` MLP outputs and an output map onto ``n_points`` grid
    points per curve."""
    n_params = len(eos_parameters)
    sizes = [n_params, units, units, 2 * n_components + 1]
    weights = [rng.normal(size=shape) for shape in zip(sizes[:-1], sizes[1:])]
    biases = [rng.normal(size=n_out) for n_out in sizes[1:]]
    metadata = {
        "activation": "silu",
        "mass_grid": {"n_points": n_points, "m_min": 1.0, "power": 2.0},
        "eos_parameters": list(eos_parameters),
    }
    return EmulatorFile(
        metadata,
        weights,
        biases,
        in_kernel=rng.normal(size=(n_params, n_params)),
        in_bias=rng.normal(size=n_params),
        out_kernel=rng.normal(size=(sizes[-1], 2 * n_points + 1)),
        out_bias=rng.normal(size=2 * n_points + 1),
    )


class TestEmulatorFile:
    seed = 42
    eos_parameters = ["a", "b", "c"]
    units = 8
    n_points = 6
    n_components = 2
    file_name = "emulator.v1"

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.emulator_file = random_emulator_file(
            self.rng, self.eos_parameters, self.units, self.n_points, self.n_components
        )

    def test_write_appends_suffix(self, tmp_path) -> None:
        path = self.emulator_file.write(tmp_path / self.file_name)
        assert path.name == f"{self.file_name}.npz"

    def test_round_trip(self, tmp_path) -> None:
        path = self.emulator_file.write(tmp_path / self.file_name)
        loaded = EmulatorFile.read(path)
        assert loaded.metadata == self.emulator_file.metadata
        for name in ("weights", "biases"):
            for array, expected in zip(
                getattr(loaded, name), getattr(self.emulator_file, name), strict=True
            ):
                assert array == pytest.approx(expected)
        for name in EmulatorFile.affine_maps:
            assert getattr(loaded, name) == pytest.approx(
                getattr(self.emulator_file, name)
            )


class TestTOVEmulator:
    seed = 42
    eos_parameters = ["a", "b", "c"]
    units = 8
    n_points = 6
    n_components = 2
    n_samples = 5

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.emulator_file = random_emulator_file(
            self.rng, self.eos_parameters, self.units, self.n_points, self.n_components
        )
        self.params = self.rng.normal(size=(self.n_samples, len(self.eos_parameters)))
        self.emulator = TOVEmulator.from_file(self.emulator_file)

    def test_single_sample_matches_batch(self) -> None:
        single = self.emulator.predict(self.params[0])
        assert single == pytest.approx(self.emulator.predict(self.params)[0])

    def test_from_npz_matches_from_file(self, tmp_path) -> None:
        path = self.emulator_file.write(tmp_path / "emulator")
        loaded = TOVEmulator.from_npz(path)
        assert loaded.metadata == self.emulator_file.metadata
        assert loaded.predict(self.params) == pytest.approx(
            self.emulator.predict(self.params)
        )
