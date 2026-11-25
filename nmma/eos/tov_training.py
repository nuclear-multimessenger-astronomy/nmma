"""Training of TOV emulators (torch).

The trained network is stored with ``write_emulator_npz`` (see tov_emulators) and
evaluated through the production path: an EoSGenerator around the numpy TOVEmulator,
compared to the raw curves by ``TOVDataset.test_vs_generator``.
"""

import copy
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nmma.core.utils import _to_json
from nmma.eos.eos_processing import EoSGenerator
from nmma.eos.eos_utils import (
    FeatureEncoder,
    MassGrid,
    PCABasis,
    error_summary,
    plot_performance,
    print_summary,
    relative_error,
)
from nmma.eos.tov_emulators import (
    TOVEmulator,
    decoder_arrays,
    network_layers,
    read_emulator_npz,
    write_emulator_npz,
)


@dataclass
class TrainingConfig:
    seed: int = 42
    label: str = "tov_emulator"
    eos_parameters: tuple = ()  # parameter names in input order

    # mass grid and data selection
    n_points: int = 50
    m_min: float = 1.0
    # 2: dense at M_tov (better for R), 1: uniform in m (best for Lambda)
    grid_power: float = 1.5
    grid_interpolation: str = "linear"  # or 'quadratic'
    mtov_range: tuple = (2.0, 3.0)
    test_frac: float = 0.1
    val_frac: float = 0.1

    # architecture
    n_components: int = 16  # PCA components per curve (R and ln Lambda)
    hidden_layers: int = 4
    units: int = 256
    activation: str = "silu"

    # loss
    tolerances: tuple = (0.05, 0.01, 0.01)  # R [km], ln Lambda, M_tov [M_sun]
    monotonic_weight: float = 0.0

    # optimisation
    epochs: int = 400
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 5

    # early stopping; the best weights are restored at the end
    early_stopping: bool = True
    patience: int = 60  # epochs without improvement before stopping
    min_epochs: int = 0  # early stopping cannot trigger before this epoch
    min_delta: float = 0.0  # neglect smaller improvements on the validation loss
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values):
        """Config from a (stored) dict. Unknown keys, e.g. from older versions, are
        ignored, missing keys get their current defaults.
        lists from JSON become tuples again where the field is a tuple"""
        types = {f.name: f.type for f in fields(cls)}
        contents = {key: value for key, value in values.items() if key in types}
        for k, v in contents.items():
            if types[k] is tuple and isinstance(v, list):
                contents[k] = tuple(v)
        return cls(**contents)

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), default=_to_json))


class TOVNetwork(nn.Module):
    """MLP predicting whitened PCA coefficients (+ scaled M_tov), followed by the fixed
    decoder to [R grid [km], ln(Lambda) grid, M_tov (M_sun)]. Inputs are standardised
    micro parameters."""

    torch_activations = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,  # alias
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "mish": nn.Mish,
        "gelu": nn.GELU,
    }

    def __init__(
        self,
        n_inputs,
        n_coefficients,
        decode_kernel,
        decode_bias,
        hidden_layers=4,
        units=256,
        activation="silu",
    ):
        super().__init__()
        layers = []
        n_in = n_inputs
        for _ in range(hidden_layers):
            layers += [nn.Linear(n_in, units), self.torch_activations[activation]()]
            n_in = units
        layers.append(nn.Linear(n_in, n_coefficients))
        self.mlp = nn.Sequential(*layers)
        for layer in self.linear_layers():  # glorot init
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # the decoder is fixed, not trainable
        self.register_buffer(
            "decode_kernel", torch.as_tensor(decode_kernel, dtype=torch.float32)
        )
        self.register_buffer(
            "decode_bias", torch.as_tensor(decode_bias, dtype=torch.float32)
        )

    @classmethod
    def from_npz(cls, path):
        """Trainable network with the weights of an emulator file (see tov_emulators)."""
        metadata, arrays = read_emulator_npz(path)
        weights, biases = network_layers(arrays)
        kernel, bias = decoder_arrays(arrays)
        network = cls(
            weights[0].shape[0],
            weights[-1].shape[1],
            kernel,
            bias,
            len(weights) - 1,
            weights[0].shape[1],
            metadata["activation"],
        )
        with torch.no_grad():
            for layer, W, b in zip(network.linear_layers(), weights, biases):
                layer.weight.copy_(torch.as_tensor(W.T))
                layer.bias.copy_(torch.as_tensor(b))
        return network

    def linear_layers(self):
        return [layer for layer in self.mlp if isinstance(layer, nn.Linear)]

    def forward(self, x):
        return self.mlp(x) @ self.decode_kernel + self.decode_bias


class TOVEmulatorTrainer(object):
    """Prepare data, train, store, evaluate and plot an emulator.

    Parameters:
    dataset: TOVDataset
    config: TrainingConfig
    outdir: str or Path
    device: str, optional
        torch device; default 'cuda' if available, else 'cpu'
    """

    def __init__(self, dataset, config=None, outdir=".", device=None):
        self.config = config or TrainingConfig()
        cfg = self.config

        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.label = cfg.label
        self.seed = cfg.seed
        torch.manual_seed(self.seed)  # global seed
        self.rng = np.random.default_rng(self.seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.prepare_dataset(cfg, dataset)
        self.split_dataset(cfg)
        self.regularise_dataset(cfg)

    def prepare_dataset(self, cfg, dataset):
        self.grid = MassGrid(cfg.n_points, cfg.m_min, cfg.grid_power)
        self.dataset = dataset
        self.indices, self.targets = dataset.prepare_targets(self.grid, cfg.mtov_range)

    def split_dataset(self, cfg):
        # explicit split; positions refer to self.indices
        order = self.rng.permutation(len(self.indices))
        n_test = int(round(cfg.test_frac * len(order)))
        n_val = int(round(cfg.val_frac * len(order)))
        self.split = {
            "test": np.sort(order[:n_test]),
            "val": np.sort(order[n_test : n_test + n_val]),
            "train": np.sort(order[n_test + n_val :]),
        }
        if cfg.test_frac == 0.0:
            self.split["test"] = self.split["train"]

    def regularise_dataset(self, cfg):
        train = self.split["train"]
        params = self.dataset.params[self.indices]
        self.encoder = FeatureEncoder().fit(params[train])
        self.features = self.encoder.transform(params)

        use_rads = self.targets[:, : cfg.n_points]
        use_ln_lams = self.targets[:, cfg.n_points : 2 * cfg.n_points][train]
        use_mtovs = self.targets[:, -1][train]
        self.basis_radius = PCABasis(cfg.n_components).fit(use_rads[train])
        self.basis_lam = PCABasis(cfg.n_components).fit(use_ln_lams)
        self.mtov_mean, self.mtov_std = use_mtovs.mean(), use_mtovs.std()

        self.model = None
        self.characteristics = None
        self._report_basis()

    def output_path(self, suffix):
        "Path of an output file of this run, e.g. output_path('_config.json')"
        return self.outdir / f"{self.label}{suffix}"

    def _report_basis(self):
        "Reconstruction error of the PCA truncation on the validation set (lower bound for the emulator)"
        n = self.config.n_points
        val = self.targets[self.split["val"]]
        for name, basis, block in [
            ("R [km]", self.basis_radius, val[:, :n]),
            ("ln Lambda", self.basis_lam, val[:, n : 2 * n]),
        ]:
            rms = np.sqrt(np.mean((basis.decode(basis.encode(block)) - block) ** 2))
            print(
                f"PCA {name}:\n \t {basis.n_components} components,  \n"
                f" \t explained variance {basis.explained:.8f},  \n"
                f" \t validation reconstruction rms {rms:.2e}"
            )

    ### training
    def train(self, evaluate=True, plot=True):
        """Train, then optionally evaluate on the test set and store the performance plot"""
        x_train, y_train, x_val, y_val = self.prepare_training()
        self.run_training(x_train, y_train, x_val, y_val)
        if evaluate:
            self.evaluate()
            if plot:
                self.visualise()
        return self.model

    def prepare_training(self):
        """Prepare the training setup."""
        cfg = self.config
        self.model = self.build_model()
        self.history = {
            key: []
            for key in (
                "loss",
                "val_loss",
                "val_rad_mse",
                "val_lnlam_mse",
                "val_mtov_mse",
                "lr",
            )
        }
        self.set_optimizer()

        x_train, y_train = self._tensors("train")
        x_val, y_val = self._tensors("val")
        steps_per_epoch = int(np.ceil(len(x_train) / cfg.batch_size))
        self.set_scheduler(steps_per_epoch)
        return x_train, y_train, x_val, y_val

    def build_model(self):
        cfg = self.config
        kernel, bias = decoder_arrays(self.scaling_arrays())
        network = TOVNetwork(
            self.features.shape[1],
            kernel.shape[0],
            kernel,
            bias,
            cfg.hidden_layers,
            cfg.units,
            cfg.activation,
        )
        return network.to(self.device)

    def set_optimizer(self, optimizer=None):
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            cfg = self.config
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

    def _tensors(self, split):
        positions = self.split[split]
        return (
            torch.as_tensor(self.features[positions], device=self.device),
            torch.as_tensor(
                self.targets[positions], dtype=torch.float32, device=self.device
            ),
        )

    def set_scheduler(self, steps_per_epoch):
        """Learning-rate schedule, stepped after every batch: linear warmup from lr/10 to lr over
        warmup_epochs, then cosine decay to lr/100 at the end of the last epoch"""
        cfg = self.config
        warmup = cfg.warmup_epochs * steps_per_epoch
        decay = max(cfg.epochs * steps_per_epoch - warmup, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=decay, eta_min=0.01 * cfg.learning_rate
        )
        if not warmup:  # SequentialLR needs at least one milestone
            self.scheduler = cosine
            return
        linear = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [linear, cosine], milestones=[warmup]
        )

    def run_training(self, x_train, y_train, x_val, y_val):
        """Train with AdamW, a warmup + cosine learning-rate schedule and optional early stopping
        on the validation loss (restoring the best weights)."""
        cfg = self.config
        best_loss, best_epoch, best_state, wait = np.inf, -1, None, 0
        generator = torch.Generator().manual_seed(self.seed)
        start = time.time()

        for epoch in range(cfg.epochs):
            self.model.train()
            order = torch.randperm(len(x_train), generator=generator).to(self.device)
            total = 0.0
            for begin in range(0, len(order), cfg.batch_size):
                batch = order[begin : begin + cfg.batch_size]
                loss = self.loss(self.model(x_train[batch]), y_train[batch])
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total += loss.item() * len(batch)

            self.model.eval()
            with torch.no_grad():
                val_loss, blocks = self.loss(self.model(x_val), y_val, per_block=True)
            val_loss = val_loss.item()
            for key, value in zip(
                self.history,
                [
                    total / len(x_train),
                    val_loss,
                    *(b.item() for b in blocks),
                    self.scheduler.get_last_lr()[0],
                ],
            ):
                self.history[key].append(value)

            if val_loss < best_loss - cfg.min_delta:
                best_loss, best_epoch, wait = val_loss, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            elif epoch >= cfg.min_epochs:
                wait += 1
            self.print_progress(epoch, time.time() - start, wait)
            if cfg.early_stopping and wait >= cfg.patience:
                print(
                    f"Early stopping after epoch {epoch + 1}: no improvement for {cfg.patience} epochs"
                )
                break

        if cfg.early_stopping and best_state is not None:
            self.model.load_state_dict(best_state)
            print(
                f"Restored best weights from epoch {best_epoch + 1} (val_loss {best_loss:.4g})"
            )
        print(f"Trained in {time.time() - start:.0f} s")
        return self.model

    def loss(self, pred, true, per_block=False):
        """
        Sum of mean squared errors of R [km], ln(Lambda) and M_tov [M_sun].
        Each divided by the squared tolerance, so the terms are balanced in
        physical terms.
        monotonic_weight > 0 additionally penalises increasing ln(Lambda) along the mass grid.
        per_block: also return the unweighted mean squared errors of the three blocks"""
        cfg = self.config
        n = cfg.n_points
        tol_r, tol_l, tol_m = cfg.tolerances
        squared = (pred - true) ** 2
        blocks = (
            squared[:, :n].mean(),
            squared[:, n : 2 * n].mean(),
            squared[:, -1].mean(),
        )
        loss = blocks[0] / tol_r**2 + blocks[1] / tol_l**2 + blocks[2] / tol_m**2
        if cfg.monotonic_weight:
            increase = torch.relu(pred[:, n + 1 : 2 * n] - pred[:, n : 2 * n - 1])
            loss = loss + cfg.monotonic_weight * (increase**2).mean() / tol_l**2
        if per_block:
            return loss, blocks
        return loss

    def print_progress(self, epoch, elapsed, no_improvement):
        """One-line progress report that fits the terminal width.
        loss and val_loss are green if they decreased since the previous epoch
        and red if they increased (only on a terminal)."""
        history = self.history
        val_loss = f"{history['val_loss'][-1]:.4g}"
        loss = f"{history['loss'][-1]:.4g}"
        if sys.stdout.isatty():
            code = 31 if no_improvement else 32
            val_loss, loss = (f"\033[{code}m{t}\033[0m" for t in (val_loss, loss))

        prefix = f"epoch {epoch + 1:4d}/{self.config.epochs}: "
        head = f"{prefix}loss {loss}, val_loss {val_loss}"
        head_width = len(head)
        blocks = (
            f" (R {history['val_rad_mse'][-1]:.2e} km^2, lnLam {history['val_lnlam_mse'][-1]:.2e}, "
            f"M_tov {history['val_mtov_mse'][-1]:.2e} M_sun^2)"
        )
        lr = f", lr {history['lr'][-1]:.2e}"
        tail = f", {elapsed:.0f} s"

        width = shutil.get_terminal_size().columns
        for optional in ((blocks, lr), (lr,), ()):
            if head_width + sum(map(len, optional)) + len(tail) < width or not optional:
                print(head + "".join(optional) + tail)
                return

    ### storage
    def emulator_content(self):
        "Metadata and arrays of the emulator file (see tov_emulators)"
        arrays = self.scaling_arrays()
        for i, layer in enumerate(self.model.linear_layers()):
            arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().astype(np.float64).T
            arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float64)
        metadata = {
            "activation": self.config.activation,
            "mass_grid": self.grid.config(),
            "eos_parameters": list(self.config.eos_parameters) or None,
            "label": self.label,
            "config": self.config.to_dict(),
            "framework": "torch",
            "data_fingerprint": self.dataset.fingerprint(),
            "output_layout": "[R grid [km], ln(Lambda) grid, M_tov [M_sun]]",
        }
        if self.characteristics is not None:
            metadata["evaluation_summary"] = self.characteristics["summary"]
        return metadata, arrays

    def scaling_arrays(self):
        "Input standardisation, PCA bases and M_tov scaling, as stored in the emulator file"
        arrays = {
            "feature_mean": self.encoder.mean,
            "feature_std": self.encoder.std,
            "feature_bounds": self.encoder.bounds,
            "mtov_mean": np.asarray(self.mtov_mean),
            "mtov_std": np.asarray(self.mtov_std),
        }
        for name, basis in (("radius", self.basis_radius), ("lam", self.basis_lam)):
            arrays |= {
                f"{name}_mean": basis.mean,
                f"{name}_components": basis.components,
                f"{name}_scales": basis.scales,
            }
        return arrays

    def save(self):
        "Store the emulator file and a copy of its metadata as {label}_config.json"
        metadata, arrays = self.emulator_content()
        path = write_emulator_npz(self.output_path(".npz"), metadata, arrays)
        with self.output_path("_config.json").open("w") as f:
            json.dump(metadata, f, indent=2, default=_to_json)
        print(f"Stored emulator at {path}")
        return path

    def load_model(self):
        "Reload the network stored by save(), e.g. to evaluate without retraining"
        self.model = TOVNetwork.from_npz(self.output_path(".npz")).to(self.device)
        self.model.eval()
        return self.model

    ### evaluation
    def emulator(self, dtype=np.float64):
        "Numpy TOVEmulator of the current network"
        return TOVEmulator.from_arrays(*self.emulator_content(), dtype=dtype)

    def generator(self, dtype=np.float64):
        "EoSGenerator around the current network, as used in production"
        names = list(self.config.eos_parameters)
        eos_parameters = names or [f"p{j}" for j in range(self.dataset.params.shape[1])]
        return EoSGenerator(self.emulator(dtype), eos_parameters)

    def evaluate(
        self,
        ref_masses=(1.4, 1.6, 1.8, 2.0),
        split="test",
        save=True,
        timing=True,
        n_calls=1000,
    ):
        """Compare the emulator, through TOVDataset.test_vs_generator, to the raw solver
        curves at reference masses. Also reports the error floors of the mass grid and of
        the PCA truncation (both using the true M_tov), so emulator errors can be attributed.
        timing: add single-call timings of speed_check (n_calls each) as summary['timing_us']"""
        generator = self.generator()
        n = self.config.n_points
        positions = self.split[split]
        data = self.dataset.subset(self.indices[positions])
        targets = self.targets[positions]
        ref_masses = np.asarray(ref_masses, dtype=float)
        characteristics = data.test_vs_generator(generator, ref_masses)
        summary = characteristics["summary"]

        # error floors
        m_tov_true = targets[:, -1]
        truths = data.reference_values(ref_masses)
        floors = {}
        for name, radius, ln_lam in [
            ("grid_floor", targets[:, :n], targets[:, n : 2 * n]),
            (
                "pca_floor",
                self.basis_radius.decode(self.basis_radius.encode(targets[:, :n])),
                self.basis_lam.decode(self.basis_lam.encode(targets[:, n : 2 * n])),
            ),
        ]:
            floors[name] = generator.mass_grid.interpolate(
                ref_masses, m_tov_true, radius, ln_lam
            )
        for j, mass in enumerate(ref_masses):
            for k, obs in enumerate(("rad", "lam")):
                truth = truths[k][:, j]
                for name, values in floors.items():
                    summary[str(mass)][obs][name] = error_summary(
                        relative_error(obs, values[k][:, j], truth), truth
                    )
        if timing:
            summary["timing_us"] = self.speed_check(generator, n_calls)

        print_summary(summary)
        self.characteristics = characteristics
        if save:
            with self.output_path("_evaluation.json").open("w") as f:
                json.dump(characteristics, f, indent=2, default=_to_json)
        return characteristics

    def speed_check(self, generator=None, n_calls=1000):
        """Time single-sample calls (the production use case) on test samples: the numpy
        emulator in float64 and float32, the generator curves, the interpolation and the
        torch network. Returns microseconds per call."""
        generator = generator or self.generator()
        emulator32 = self.emulator(np.float32)
        test = self.split["test"]
        pick = np.random.default_rng(0).integers(len(test), size=n_calls)
        params = self.dataset.params[self.indices[test[pick]]][:, None, :]
        features = torch.as_tensor(self.features[test[pick]], device=self.device)[
            :, None, :
        ]
        ref_masses = np.array([1.4, 1.6, 1.8, 2.0])
        timings = {}

        def time_calls(name, func, inputs):
            func(inputs[0])  # warm-up
            start = time.perf_counter()
            for x in inputs:
                func(x)
            timings[name] = (time.perf_counter() - start) / len(inputs) * 1e6

        time_calls("emulator predict", generator.emulator.predict, params)
        time_calls("emulator predict (float32)", emulator32.predict, params)
        time_calls(
            "generator curves", lambda x: generator(generator.parameter_dict(x)), params
        )
        radii, ln_lams, m_tov = generator(generator.parameter_dict(params[0]))
        time_calls(
            "interpolate (4 masses)",
            lambda _: generator.mass_grid.interpolate(
                ref_masses, m_tov, radii, ln_lams
            ),
            params,
        )
        if self.model is not None:
            self.model.eval()
            with torch.inference_mode():
                time_calls("torch network", self.model, features)
        return timings

    ### plotting
    def visualise(self, characteristics=None, split="test", **plot_kwargs):
        """Store the performance overview (plot_performance) as {outdir}/{label}_performance.pdf;
        evaluates first if necessary"""
        characteristics = (
            characteristics or self.characteristics or self.evaluate(split=split)
        )
        test_data = self.dataset.subset(self.indices[self.split[split]])
        path = self.output_path("_performance.pdf")
        fig = plot_performance(
            self.generator(), test_data, characteristics, save=path, **plot_kwargs
        )
        print(f"Stored {path}")
        return fig
