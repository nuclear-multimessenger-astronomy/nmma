import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import erf

from nmma.core.utils import _to_json


def _silu(x: np.ndarray) -> np.ndarray:
    # tanh form of x * sigmoid(x), avoids overflow warnings of exp(-x)
    return 0.5 * x * (1.0 + np.tanh(0.5 * x))


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, 0.01 * x)


def _mish(x: np.ndarray) -> np.ndarray:
    return x * np.tanh(np.logaddexp(0.0, x))


def _linear(x: np.ndarray) -> np.ndarray:
    return x


ACTIVATIONS = {
    "silu": _silu,
    "swish": _silu,  # alias
    "tanh": np.tanh,
    "relu": _relu,
    "leaky_relu": _leaky_relu,
    "mish": _mish,
    "gelu": _gelu,
    "linear": _linear,
}


@dataclass
class EmulatorFile:
    """Content of an emulator file (see module docstring): the MLP layers between a
    fixed affine input and output map, and the metadata.

    Parameters
    ----------
    metadata: dict
        JSON-serialisable metadata, at least the hidden-layer "activation".
    weights, biases: lists of arrays
        Per MLP layer, shapes (n_in, n_out) and (n_out,).
    in_kernel, in_bias: arrays
        Affine map from the raw parameters to the MLP input.
    out_kernel, out_bias: arrays
        Affine map from the MLP output to the emulator output.
    """

    metadata: dict
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    in_kernel: np.ndarray
    in_bias: np.ndarray
    out_kernel: np.ndarray
    out_bias: np.ndarray

    affine_maps = ("in_kernel", "in_bias", "out_kernel", "out_bias")

    @classmethod
    def read(cls, path: str | Path) -> "EmulatorFile":
        """Content of a file written by ``write``."""
        with np.load(path) as data:
            metadata = json.loads(str(data["meta"]))
            n_layers = sum(key.startswith("W") for key in data.files)
            weights = [data[f"W{i}"] for i in range(n_layers)]
            biases = [data[f"b{i}"] for i in range(n_layers)]
            maps = [data[name] for name in cls.affine_maps]
        return cls(metadata, weights, biases, *maps)

    def write(self, path: str | Path) -> Path:
        """Store as .npz (the suffix is appended if missing); returns the path."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(f"{path.name}.npz")
        layers = {f"W{i}": weights for i, weights in enumerate(self.weights)}
        layers |= {f"b{i}": bias for i, bias in enumerate(self.biases)}
        maps = {name: getattr(self, name) for name in self.affine_maps}
        # stored as a plain unicode array, so loading needs no pickle
        meta = np.array(json.dumps(self.metadata, default=_to_json))
        np.savez(path, meta=meta, **layers, **maps)
        return path


class TOVEmulator:
    """Fast numpy evaluation of a trained TOV emulator.

    Parameters
    ----------
    weights, biases: lists of arrays
        Per layer, shapes (n_in, n_out) and (n_out,), with all scaling folded in: the
        network maps raw micro parameters to [R grid [km], ln(Lambda) grid, M_tov].
    activation: str
        Hidden-layer activation, key of ACTIVATIONS.
    metadata: dict, optional
        Metadata of the emulator file, e.g. the mass grid configuration.
    dtype: numpy dtype
        Precision of the evaluation.
    """

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activation: str,
        metadata: dict | None = None,
        dtype: type = np.float64,
    ) -> None:
        self.dtype = dtype
        self.weights = [np.ascontiguousarray(W, dtype=dtype) for W in weights]
        self.biases = [np.ascontiguousarray(b, dtype=dtype) for b in biases]
        self.activation = activation
        self._act = ACTIVATIONS[activation]
        self.metadata = metadata or {}

    @classmethod
    def from_file(cls, file: EmulatorFile, dtype: type = np.float64) -> "TOVEmulator":
        """Emulator of an emulator file's content, with its affine input and output
        maps folded into the first and last layer."""
        weights = [np.asarray(W, dtype=float) for W in file.weights]
        biases = [np.asarray(b, dtype=float) for b in file.biases]
        biases[0] = file.in_bias @ weights[0] + biases[0]
        weights[0] = file.in_kernel @ weights[0]
        biases[-1] = biases[-1] @ file.out_kernel + file.out_bias
        weights[-1] = weights[-1] @ file.out_kernel
        return cls(weights, biases, file.metadata["activation"], file.metadata, dtype)

    @classmethod
    def from_npz(cls, path: str | Path, dtype: type = np.float64) -> "TOVEmulator":
        """Emulator from a file written by ``EmulatorFile.write``."""
        return cls.from_file(EmulatorFile.read(path), dtype)

    def predict(self, params: np.ndarray) -> np.ndarray:
        """Network output [R grid, ln(Lambda) grid, M_tov] for micro parameters of
        shape (..., n_params)."""
        h = np.asarray(params, dtype=self.dtype)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = self._act(h @ W + b)
        return h @ self.weights[-1] + self.biases[-1]
