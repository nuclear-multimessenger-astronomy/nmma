import os
from argparse import Namespace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..core.conversion import (
    MassGrid,
    eos_to_system_parameters,
    max_mass_increase_by_spin,
    non_rot_mass_from_spin,
)
from .eos_utils import TOVDataset
from .tov_emulators import TOVEmulator


class EoSGenerator:
    """Maps micro-EOS parameters to radius and ln Lambda curves on a mass grid.

    The native output of a generator (``curves``) are the radius and ln Lambda curves
    on the mass grid of each EOS, stacked as (n_eos, 2, n_points) for
    ``MassGrid.interpolate``, and the TOV masses, from which the grid masses follow.
    ``__call__`` returns radii and ln Lambdas separately, and ``tables_from_curves``
    builds (radius, mass, ln Lambda) tables from them.

    The default backend is a TOVEmulator stored as .npz, whose metadata provides the
    mass grid it was trained on and, optionally, the parameter names. Use
    ``build_generator`` to pick the generator class from the arguments.

    Parameters
    ----------
    emulator: str | Path | TOVEmulator
        Emulator file (.npz) or an already loaded emulator, i.e. an object with
        ``predict`` and ``metadata`` like TOVEmulator.
    eos_parameters: list of str, optional
        Names of the parameters passed to the emulator, in input order; default
        from the emulator metadata.
    mass_grid: MassGrid | dict | int | tuple | None
        parameter from which the MassGrid can be built
    """

    def __init__(
        self,
        emulator: str | Path | TOVEmulator | None,
        eos_parameters: list[str] = None,
        mass_grid=None,
        **kwargs,
    ) -> None:
        self.set_emulator(emulator, **kwargs)
        if eos_parameters is None:
            try:
                eos_parameters = self.emulator.metadata["eos_parameters"]
            except KeyError:
                raise ValueError(f"{type(self).__name__} needs eos_parameters")
        self.eos_parameters = list(eos_parameters)

        self.build_mass_grid(mass_grid)

    def set_emulator(self, emulator: str | Path | TOVEmulator, **kwargs) -> None:
        """Load the emulator from its file, or use an already loaded one."""
        if isinstance(emulator, (str, Path)):
            emulator = TOVEmulator.from_npz(emulator)
        self.emulator = emulator

    def build_mass_grid(self, mass_grid) -> None:
        """By default the mass grid the emulator was trained on, from its metadata."""
        if mass_grid is None:
            mass_grid = self.base_mass_grid()

        if isinstance(mass_grid, MassGrid):
            self.mass_grid = mass_grid
        elif isinstance(mass_grid, dict):
            self.mass_grid = MassGrid.from_config(mass_grid)
        elif isinstance(mass_grid, (int, np.integer, tuple)):
            self.mass_grid = MassGrid.from_n_mass_samples(mass_grid)
        else:
            raise ValueError(f"Cannot initialise MassGrid from {mass_grid}")

    def base_mass_grid(self):
        return self.emulator.metadata["mass_grid"]

    @property
    def n_mass_samples(self) -> int:
        """Number of mass grid points per EOS."""
        return self.mass_grid.n_points

    def __call__(self, parameters: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Radii [km] and ln Lambdas on the mass grid, shape (n_eos, n_points), and
        TOV masses, shape (n_eos,), of one batch of EOSs; views of ``curves``."""
        curves, m_tov = self.curves(parameters)
        return curves[:, 0], curves[:, 1], m_tov

    # shared methods
    def assemble_eos_params(self, parameters: dict) -> np.ndarray:
        """Assemble the parameters for the EoS model into a 2D-array for the emulator"""
        eos_params = np.array([parameters[par] for par in self.eos_parameters]).T
        return np.atleast_2d(eos_params)

    def parameter_dict(self, params: ArrayLike) -> dict:
        """Parameter dict for an array (n_eos, n_params) in emulator input order."""
        params = np.atleast_2d(params)
        return {name: params[:, j] for j, name in enumerate(self.eos_parameters)}

    def batch_shape(self, parameters: dict) -> tuple:
        """Shape of the batch of EOSs in ``parameters``, () for a single sample."""
        return np.shape(parameters[self.eos_parameters[0]])

    # preferred, slim output
    def curves(self, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
        """Radius [km] and ln Lambda curves on the mass grid, shape
        (n_eos, 2, n_points), and TOV masses, shape (n_eos,), of one batch of EOSs."""
        return self.predict_curves(self.assemble_eos_params(parameters))

    def predict_curves(self, eos_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Curves and TOV masses for the assembled parameters, see ``curves``."""
        return self.split_prediction(self.predict(eos_params))

    # alternative table-like output
    def tables_from_curves(self, curves: np.ndarray, m_tov: np.ndarray) -> np.ndarray:
        """(radius, mass, ln Lambda) tables of ``curves``, shape (n_eos, 3, n_points),
        rebuilding the masses from M_tov."""
        masses = self.mass_grid.masses(m_tov)[:, None]
        return np.concatenate([curves[:, :1], masses, curves[:, 1:]], axis=1)

    # specific methods
    def predict(self, eos_params: np.ndarray) -> np.ndarray:
        """Raw emulator output for the assembled parameters."""
        return self.emulator.predict(eos_params)

    def split_prediction(self, predictions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Split the flat output [R grid | ln(Lambda) grid | M_tov] into the curves
        (n_eos, 2, n_points), a view of ``predictions``, and the TOV masses."""
        predictions = np.asarray(predictions)
        n = self.n_mass_samples
        return predictions[:, : 2 * n].reshape(-1, 2, n), predictions[:, -1]


class KerasEoSGenerator(EoSGenerator):
    """Older Keras emulators with output [R grid | log10(Lambda) grid | M_tov].
    As the model file carries no metadata, the mass grid must be given explicitly.

    Parameters
    ----------
    emulator: str | Path
        Path to the .keras model.
    eos_parameters: list of str
        Names of the parameters passed to the emulator, in input order.
    mass_grid: MassGrid | dict | int | tuple, optional
        The mass grid of the emulator, or its config; an int or tuple is the
        former ``n_mass_samples`` configuration.
    backend: str, optional
        Expected Keras backend, asserted against the active one.
    """

    # factors from the radius and log10(Lambda) curves to radius and ln(Lambda)
    curve_units = np.array([[1.0], [np.log(10)]])

    def set_emulator(
        self, emulator: str | Path, backend: str | None = None, **kwargs
    ) -> None:
        """Load the Keras model and choose the call best suited for small batches."""
        import keras as k

        if backend:
            assert k.backend.backend() == backend, (
                f"Keras Backend mismatch: {k.backend.backend()} vs {backend}. "
                f"please set the environment variable KERAS_BACKEND to {backend}"
            )
        self.emulator = k.saving.load_model(
            emulator, custom_objects=None, compile=False
        )

        # NOTE: in sampling, the tensorflow-predict method is much slower
        #  than pure __call__ as it tries to cater to larger batches!
        if k.backend.backend() == "tensorflow":
            self.predict = self.tensorflow_predict
        elif k.backend.backend() == "jax":
            self.predict = self.jax_predict

    def base_mass_grid(self):
        return 30

    def jax_predict(self, x: np.ndarray) -> np.ndarray:
        """``self.emulator.predict(x, verbose=0)``, for the jax Keras backend."""
        return self.emulator.predict(x, verbose=0)

    def tensorflow_predict(self, x: np.ndarray) -> np.ndarray:
        """``self.emulator(x)`` -- faster than ``.predict`` for the small
        batches used during sampling, on the tensorflow Keras backend."""
        return self.emulator(x)

    def split_prediction(self, predictions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """As EoSGenerator.split_prediction, converting log10(Lambda) to ln."""
        predictions = np.asarray(predictions, dtype=float)
        curves, m_tov = super().split_prediction(predictions)
        return curves * self.curve_units, m_tov


class SklearnEoSGenerator(EoSGenerator):
    """Chiral-EFT low-energy-constant EOS emulator: separate sklearn-style
    models (and scalers) for mass, radius, and lambda, rather than a combined model.

    Parameters
    ----------
    emulator: None
        Unused, the models are given as keyword arguments.
    eos_parameters: list of str
        Names of the parameters passed to the emulator, in input order.
    mass_grid: MassGrid | dict | int | tuple, optional
        The mass grid of the emulator, default 30 equally spaced masses.
    kwargs:
        Paths to joblib-pickled "feature_scaler", "lambda_scaler",
        "radius_scaler", "mass_emulator", "radius_emulator" and
        "lambda_emulator".
    """

    def set_emulator(self, emulator: None, **kwargs) -> None:
        """Load the scalers and the three emulators."""
        import joblib

        self.feature_scaler = joblib.load(kwargs["feature_scaler"])
        self.lambda_scaler = joblib.load(kwargs["lambda_scaler"])
        self.radius_scaler = joblib.load(kwargs["radius_scaler"])

        self.mass_emulator = joblib.load(kwargs["mass_emulator"])
        self.radius_emulator = joblib.load(kwargs["radius_emulator"])
        self.lambda_emulator = joblib.load(kwargs["lambda_emulator"])

    def base_mass_grid(self):
        return 30

    def predict(self, parameters):
        """Scale ``parameters`` (the assembled 2D parameter
        array from ``assemble_eos_params``) and run the mass, radius,
        and lambda emulators on it.

        Returns
        -------
        tuple
            (mass_prediction, radius_prediction, lambda_prediction).
        """
        # Scale the input features
        scaled_features = self.feature_scaler.transform(parameters)

        # Make predictions using the emulators
        mass_prediction = self.mass_emulator.predict(scaled_features)
        radius_prediction = self.radius_emulator.predict(scaled_features)
        lambda_prediction = self.lambda_emulator.predict(scaled_features)

        return (mass_prediction, radius_prediction, lambda_prediction)

    def split_prediction(self, predictions: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-transform the (TOV mass, radius, log10 lambda) predictions via
        their scalers into the curves and TOV masses, see ``curves``."""
        mass_data, rad_data, lam_data = predictions
        radii = self.radius_scaler.inverse_transform(rad_data)
        ln_lams = np.log(10) * self.lambda_scaler.inverse_transform(lam_data)
        return np.stack([radii, ln_lams], axis=1), np.ravel(mass_data)


class TabulatedEoSGenerator(EoSGenerator):
    """Curves of tabulated EOSs, selected by the 0-based "EOS" parameter.

    Every table is cut at its maximum mass and resampled onto ``mass_grid`` once at
    initialisation, so the curves come in the same format as those of the
    emulators. The raw tables remain available through ``raw_tables`` and
    ``reference_dataset``, for validation and plotting.

    Parameters
    ----------
    args: Namespace
        ``eos_file`` for a single table, or ``eos_data`` (and ``Neos``), see
        ``read_eos_files``. Tables have columns radius [km], mass [M_sun], Lambda
        (or ln Lambda).
    mass_grid: MassGrid | dict, optional
        Grid for resampling, default ``MassGrid()``. Its ``m_min`` must lie within
        every table, and below every mass the EOSs are evaluated at.
    """

    def __init__(
        self,
        args: Namespace | None = None,
        mass_grid: MassGrid | dict | None = None,
    ) -> None:
        self.eos_parameters = ["EOS"]
        if getattr(args, "eos_file", None):
            self.eos_files = [Path(args.eos_file)]
        else:
            self.read_eos_files(args.eos_data, getattr(args, "Neos", None))
        self.Neos = len(self.eos_files)

        self.build_mass_grid(mass_grid or MassGrid())
        curves, m_tov = zip(*(self.resample_table(f) for f in self.eos_files))
        self.resampled_curves, self.m_tov = np.array(curves), np.array(m_tov)

    def batch_shape(self, parameters: dict) -> tuple:
        """Shape of the batch of EOSs in ``parameters``, () for a single sample;
        without "EOS", a single table serves every entry of ``parameters``."""
        if "EOS" in parameters or self.Neos > 1:
            return np.shape(parameters["EOS"])
        try:
            return np.shape(next(iter(parameters.values()), 0))
        except TypeError:  # this fails for pandasDF
            return (len(parameters),)

    def assemble_eos_params(self, parameters: dict) -> np.ndarray:
        return self.selected_indices(parameters)[:, None]

    def predict_curves(self, eos_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The resampled curves of the EOS indices ``eos_params``, see ``curves``."""
        indices = np.ravel(eos_params).astype(int)
        return self.resampled_curves[indices], self.m_tov[indices]

    def resample_table(self, eos_file: str | Path) -> tuple[np.ndarray, float]:
        """Radius and ln Lambda curve of a table on the mass grid, shape
        (2, n_points), and its TOV mass."""
        radius, mass, ln_lam = self.read_eos_table(eos_file)
        mtov = mass[-1]
        if mtov < self.mass_grid.m_min:
            raise ValueError(
                f"EOS {eos_file} has maximum mass {mass[-1]:.3f} < "
                f"grid minimum mass {self.mass_grid.m_min:.3f}"
            )
        elif mass[0] > self.mass_grid.m_min:
            raise ValueError(
                f"EOS {eos_file} has minimum mass {mass[0]:.3f} > "
                f"grid minimum mass {self.mass_grid.m_min:.3f}"
            )
        return self.mass_grid.resample(mass, radius, ln_lam), mtov

    def raw_tables(self, parameters: dict) -> list[np.ndarray]:
        """The raw (radius, mass, ln Lambda) tables of the requested EOSs, read
        from file once per distinct EOS."""
        indices = self.selected_indices(parameters)
        unique, inverse = np.unique(indices, return_inverse=True)
        tables = [self.read_eos_table(self.eos_files[i]) for i in unique]
        return [tables[j] for j in np.ravel(inverse)]

    def read_eos_table(
        self, eos_file, usecols=[0, 1, 2], test_scale=True
    ) -> np.ndarray:
        """(radius, mass, ln Lambda) of a tabulated EOS file with columns radius, mass
        and Lambda; Lambda columns with values above 100 are taken as linear."""
        # TODO: detecting the Lambda scale by its values is fragile
        table = np.loadtxt(eos_file, usecols=usecols, unpack=True)
        if test_scale and np.any(table[2] > 100):
            table[2] = np.log(table[2])
        return table

    def reference_dataset(self) -> TOVDataset:
        """The raw tables as TOVDataset, with the EOS index as parameter."""
        tables = [self.read_eos_table(f) for f in self.eos_files]
        length = max(table.shape[1] for table in tables)
        radius, mass, ln_lam = np.stack(
            [
                np.pad(t, ((0, 0), (0, length - t.shape[1])), constant_values=np.nan)
                for t in tables
            ],
            axis=1,
        )
        indices = np.arange(self.Neos, dtype=float)[:, None]
        return TOVDataset(indices, mass, radius, np.exp(ln_lam))

    def selected_indices(self, parameters: dict) -> np.ndarray:
        """Indices of the requested EOSs, shape (n_eos,); a single table serves
        every entry."""
        if self.Neos == 1:
            n_entries = int(np.prod(self.batch_shape(parameters)))
            return np.zeros(n_entries, dtype=int)
        return np.ravel(parameters["EOS"]).astype(int)

    def read_eos_files(self, eos_data: str | Path, Neos: int | None) -> list[Path]:
        """Files ``1..Neos`` of a directory of ``{j}.dat`` files, or of a glob pattern
        whose matches share a common prefix and suffix around the EOS number."""
        eos_path = Path(eos_data)
        if eos_path.is_dir():
            pre, suf, pattern = "", ".dat", "*.dat"
        else:
            eos_path, pattern = eos_path.parent, eos_path.name
            names = [f.name for f in sorted(eos_path.glob(pattern))]
            if not names:
                raise ValueError(f"No files match {eos_path}/{pattern}")
            pre = os.path.commonprefix(names)
            suf = os.path.commonprefix([name[::-1] for name in names])[::-1]
        self.Neos = Neos or len(list(eos_path.glob(pattern)))
        self.eos_files = [eos_path / f"{pre}{j + 1}{suf}" for j in range(self.Neos)]


def build_generator(args: Namespace) -> EoSGenerator:
    """The generator described by ``args``: tabulated EOSs for ``eos_file`` or
    ``eos_data``, otherwise an emulator from ``emulator_path``, ``micro_eos_model``
    and ``emulator_metadata`` (whose entries take precedence)."""
    if getattr(args, "eos_file", None) or getattr(args, "eos_data", None):
        return TabulatedEoSGenerator(args)

    metadata = dict(args.emulator_metadata)
    for key in ("emulator_path", "micro_eos_model"):
        if key not in metadata:
            metadata[key] = getattr(args, key)
    emulator_path = metadata.pop("emulator_path", None)
    eos_model_type = str(metadata.pop("micro_eos_model", "")).lower()
    if "n_mass_samples" in metadata:
        metadata.setdefault("mass_grid", metadata.pop("n_mass_samples"))

    legacy_micro_parameters = {
        "nep-5": ["K_sat", "L_sym", "K_sym", "3n_sat", "5n_sat"],
        # 6 chiral-EFT low-energy constants
        "lec-7": ["d11", "d22", "d3", "d4", "d6", "d7"],
        # + 7 saturation/speed-of-sound parameters
        "lec-13": [
            "d11", "d22", "d3", "d4", "d6", "d7",
            "ksat", "qsat", "zsat", "cssq1", "cssq2", "cssq3", "cssq4",
        ],
    }  # fmt: skip
    if eos_model_type in legacy_micro_parameters:
        metadata["eos_parameters"] = legacy_micro_parameters[eos_model_type]

    if emulator_path is None:
        if "feature_scaler" in metadata:
            return SklearnEoSGenerator(None, **metadata)
        raise ValueError("emulator_path must be provided in args or metadata")

    suffix = Path(emulator_path).suffix
    if suffix == ".npz":
        return EoSGenerator(emulator_path, **metadata)
    elif suffix == ".keras":
        return KerasEoSGenerator(emulator_path, **metadata)
    else:
        raise ValueError(f"Unsupported emulator file {emulator_path}")


class EoSConverter:
    """Parameter conversion from sampled EOS (and binary) parameters to macroscopic
    NS properties: TOV mass, TOV radius, R_14 and R_16 of each EOS and, for GW
    analyses, tidal deformability, radius and NS-flag of each component.

    The curves of the last converted batch are kept as ``curves``
    (n_eos, 2, n_points), with views ``radii`` and ``ln_lams``, and ``m_tov``
    (n_eos,), and are evaluated at any mass by ``curves_at`` along the mass grid of
    the generator. Results have the shape of
    the sampled parameters: numpy scalars for a single sample.

    Parameters
    ----------
    args: argparse.Namespace, optional
        Arguments for ``build_generator``, and ``spin_correction``.
    generator: EoSGenerator, optional
        Generator to use instead of building one from ``args``.
    macro_only: bool
        Whether ``parameter_conversion`` derives only the properties of the EOS,
        e.g. for EOS-only analyses, rather than those of the components as well.
    """

    characteristic_masses = np.array([1.4, 1.6])
    # radius and ln Lambda outside the mass range of an EOS
    outside_values = np.array([0.0, -np.inf])[:, None, None]
    bh_lams = np.array([-np.inf])[None, None]
    bh_rad = np.array([0])[None, None]

    def __init__(
        self,
        args: Namespace | None = None,
        generator: EoSGenerator | None = None,
        macro_only: bool = False,
    ) -> None:
        self.spin_correction = getattr(args, "spin_correction", False)
        if generator is None:
            generator = build_generator(args)
        self.generator = generator
        self.mass_grid = generator.mass_grid
        if macro_only:
            self.parameter_conversion = self.compute_macro_parameters
        else:
            self.parameter_conversion = self.full_eos_conversion

    def __call__(self, parameters):
        """Forward to ``self.parameter_conversion``."""
        return self.parameter_conversion(parameters)

    def full_eos_conversion(self, parameters):
        """The default ``parameter_conversion`` for GW+EOS joint analyses; looks up
        the curves once for both the characteristic radii and the components."""
        self.update_curves(parameters)
        components = self.components(parameters)
        masses = np.empty((len(self.m_tov), 4))
        # masses [M_sun] of the characteristic radii R_14 and R_16
        masses[:, :2] = self.characteristic_masses
        masses[:, 2:] = components[2]
        radii, ln_lams = self.curves_at(masses)
        self.add_to(parameters, self.macro_properties(radii[:, :2]))
        system = self.system_properties(*components, radii[:, 2:], ln_lams[:, 2:])
        return self.add_to(parameters, system)

    def update_curves(self, parameters) -> None:
        """Predict and keep the curves of the EOSs in ``parameters``."""
        self.batch_shape = self.generator.batch_shape(parameters)
        self.curves, self.m_tov = self.generator.curves(parameters)
        self.radii, self.ln_lams = self.curves[:, 0], self.curves[:, 1]

    def component_pair(self, parameters, key: str) -> np.ndarray:
        """Values of ``key`` (e.g. "a_{}") for both components, shape (n_eos, 2)."""
        pair = np.empty((len(self.m_tov), 2))
        pair[:, 0] = np.ravel(parameters[key.format(1)])
        pair[:, 1] = np.ravel(parameters[key.format(2)])
        return pair

    def components(self, parameters) -> tuple:
        """Source masses and spins of both components in ``parameters``, the masses
        at which the current curves describe them, and their maximum masses, each
        of shape (n_eos, 2). With ``spin_correction``, the curves are evaluated at
        the non-rotating masses, and rigid rotation raises the maximum mass."""
        m_source = self.component_pair(parameters, "mass_{}_source")
        chi = self.component_pair(parameters, "a_{}")
        m_tov = self.m_tov[:, None]
        if not self.spin_correction:
            return m_source, chi, m_source, m_tov
        m_eval = self.non_rotating_mass(m_source, chi, m_tov)
        return m_source, chi, m_eval, m_tov * max_mass_increase_by_spin(chi)

    def add_to(self, parameters, values: dict):
        """``parameters`` with ``values`` of the current batch added in its batch
        shape, as numpy scalars for a single sample."""
        if self.batch_shape == ():
            for key, value in values.items():
                parameters[key] = value[0]
        else:
            for key, value in values.items():
                parameters[key] = np.reshape(value, self.batch_shape)
        return parameters

    def _interp_at(self, masses, array, out_vals) -> tuple[np.ndarray, np.ndarray]:
        values = self.mass_grid.interpolate(masses, self.m_tov, array)
        return np.where(np.isnan(values), out_vals, values)

    def curves_at(self, masses: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Radii and ln Lambdas of the current curves at ``masses`` (shape
        (n_masses,) for every EOS, or (n_eos, n_masses)), shape (n_eos, n_masses).
        Outside [m_min, M_tov] of each EOS, the radius is 0 and ln Lambda -inf."""
        return self._interp_at(masses, self.curves, self.outside_values)

    def ln_lam_at(self, masses: ArrayLike) -> np.ndarray:
        """ln Lambdas of the current curves at ``masses``, see ``curves_at``."""
        return self._interp_at(masses, self.ln_lams[:, None], self.bh_lams)[0]

    def rad_at(self, masses: ArrayLike) -> np.ndarray:
        """Radii of the current curves at ``masses``, see ``curves_at``."""
        return self._interp_at(masses, self.radii[:, None], self.bh_rad)[0]

    def non_rotating_mass(self, m_source, chi, m_tov, tol=1e-10, max_iter=50):
        """
        Mass of the non-rotating star with the central pressure of a star of mass
        ``m_source`` spinning with ``chi``, i.e. the solution ``m_0 <= m_tov`` of
        ``m_source = m_0 * (1 + dm_bar(m_0) * chi**2)``, see
        :func:`rotational_mass_increase`. Solved by fixed-point iteration of
        :func:`non_rot_mass_from_spin`.

        Parameters
        ----------
        m_source, chi : np.ndarray
            Masses and spins, shape ``(*batch, 2)``.
        m_tov : np.ndarray
            TOV masses, broadcastable to ``m_source``.
        tol : float, optional
            Absolute tolerance of the masses [M_sun].
        max_iter : int, optional
            Maximum number of iterations.

        Returns
        -------
        np.ndarray
            The non-rotating masses, capped at ``m_tov``. If an iterate leaves the
            mass range of the EOS, it is returned as is, so the EOS stays undefined
            there.
        """
        m_0 = np.minimum(m_source, m_tov)
        for _ in range(max_iter):
            ln_lam = self.ln_lam_at(m_0)
            defined = np.isfinite(ln_lam)

            update = non_rot_mass_from_spin(
                m_source, chi, np.where(defined, ln_lam, 0.0)
            )
            update = np.where(defined, np.minimum(update, m_tov), m_0)

            converged = np.all(np.abs(update - m_0) < tol)
            m_0 = update
            if converged:
                break
        return m_0

    def macro_properties(self, characteristic_radii: np.ndarray) -> dict:
        """TOV mass and radius of the current curves, and their radii at the
        ``characteristic_masses``, shape (n_eos, 2)."""
        return {
            "TOV_mass": self.m_tov,
            "TOV_radius": self.radii[:, -1],
            "R_14": characteristic_radii[:, 0],
            "R_16": characteristic_radii[:, 1],
        }

    def add_system_properties(self, parameters):
        """``parameters`` with lambda, radius and NS-flag of both components added,
        from the current curves."""
        components = self.components(parameters)
        radii, ln_lams = self.curves_at(components[2])
        system = self.system_properties(*components, radii, ln_lams)
        return self.add_to(parameters, system)

    def system_properties(self, m_source, chi, m_eval, m_max, radii, ln_lams) -> dict:
        """Lambda, radius and NS-flag of both components from their
        :meth:`components` and the ``radii`` and ``ln_lams`` of the curves at
        ``m_eval``, each of shape (n_eos, 2)."""
        # compactness is constant to good approximation under rotation, so the
        # radius scales with the mass; Lambda is assumed to be kept constant
        radii = radii * m_source / m_eval
        lambdas, radii, is_ns = eos_to_system_parameters(
            radii, ln_lams, m_source, chi, m_max
        )
        system = {}
        for j in range(2):
            system[f"lambda_{j + 1}"] = lambdas[:, j]
            system[f"radius_{j + 1}"] = radii[:, j]
            system[f"is_ns_{j + 1}"] = is_ns[:, j]
        return system

    def compute_macro_parameters(self, parameters):
        """Predict the curves of the EOSs in ``parameters`` and add TOV_mass,
        TOV_radius, R_14 and R_16 (0 above the TOV mass).

        Parameters
        ----------
        parameters: dict
            Must contain the parameters of the generator (e.g. "EOS").

        Returns
        -------
        dict
            ``parameters``, with TOV_mass/TOV_radius/R_14/R_16 added.
        """
        self.update_curves(parameters)
        radii, _ = self.curves_at(self.characteristic_masses)
        return self.add_to(parameters, self.macro_properties(radii))

    @property
    def macro_parameters(self) -> dict:
        """The current curves as radii, masses and ln Lambdas, each of shape
        (*batch_shape, n_points), e.g. for EoS constraints and plotting."""
        shape = self.batch_shape + (self.mass_grid.n_points,)
        tables = {
            "radii": self.radii,
            "masses": self.mass_grid.masses(self.m_tov),
            "ln_lams": self.ln_lams,
        }
        return {key: np.reshape(value, shape) for key, value in tables.items()}

    def test_vs_dataset(
        self,
        dataset: TOVDataset | None = None,
        ref_masses: ArrayLike = (1.4, 1.6, 1.8, 2.0),
    ) -> dict:
        """``dataset.test_vs_generator`` of this converter's generator; without a
        dataset, a tabulated generator compares to its own raw tables."""
        if dataset is None:
            dataset = self.generator.reference_dataset()
        return dataset.test_vs_generator(self.generator, ref_masses)
