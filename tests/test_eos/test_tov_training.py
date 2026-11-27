import numpy as np
import pytest
import torch
from test_tov_emulators import random_emulator_file

from nmma.eos.tov_emulators import ACTIVATIONS, EmulatorFile, TOVEmulator
from nmma.eos.tov_training import TOVNetwork, TrainingConfig


class TestTrainingConfig:
    def setup_method(self) -> None:
        self.config = TrainingConfig(
            epochs=15, mtov_range=(2.1, 2.9), eos_parameters=("a", "b")
        )
        self.stored_epochs = 7
        self.obsolete_entry = {"n_members": 3}

    def test_dict_round_trip(self) -> None:
        assert TrainingConfig.from_dict(self.config.to_dict()) == self.config

    def test_unknown_keys_are_ignored_and_missing_keys_get_defaults(self) -> None:
        stored = {"epochs": self.stored_epochs} | self.obsolete_entry
        config = TrainingConfig.from_dict(stored)
        assert config == TrainingConfig(epochs=self.stored_epochs)


class TestTOVNetwork:
    seed = 42
    eos_parameters = ["a", "b", "c"]
    units = 8
    n_points = 6
    n_components = 2
    n_samples = 5
    activation_range = (-5.0, 5.0, 101)

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.emulator_file = random_emulator_file(
            self.rng, self.eos_parameters, self.units, self.n_points, self.n_components
        )
        self.params = self.rng.normal(size=(self.n_samples, len(self.eos_parameters)))
        self.activation_inputs = np.linspace(*self.activation_range)

    def test_matches_numpy_emulator_of_its_file(self) -> None:
        "The torch network and the numpy emulator of its exported file agree"
        network = TOVNetwork.from_file(self.emulator_file).double()
        exported = network.emulator_file(self.emulator_file.metadata)
        with torch.no_grad():
            output = network(torch.as_tensor(self.params)).numpy()
        expected = TOVEmulator.from_file(exported).predict(self.params)
        assert output == pytest.approx(expected)

    def test_emulator_file_round_trip(self) -> None:
        network = TOVNetwork.from_file(self.emulator_file)
        exported = network.emulator_file(self.emulator_file.metadata)
        assert exported.metadata == self.emulator_file.metadata
        for name in ("weights", "biases"):
            for array, expected in zip(
                getattr(exported, name), getattr(self.emulator_file, name), strict=True
            ):
                assert array == pytest.approx(expected)
        for name in EmulatorFile.affine_maps:
            assert getattr(exported, name) == pytest.approx(
                getattr(self.emulator_file, name)
            )

    def test_torch_activations_match_numpy_activations(self) -> None:
        x = torch.as_tensor(self.activation_inputs)
        for name, numpy_activation in ACTIVATIONS.items():
            torch_output = TOVNetwork.torch_activations[name]()(x).numpy()
            expected = numpy_activation(self.activation_inputs)
            assert torch_output == pytest.approx(expected)
