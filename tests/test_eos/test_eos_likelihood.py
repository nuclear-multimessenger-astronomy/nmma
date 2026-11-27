import numpy as np
import pytest
from helpers import N_EOS_MACRO, eos_macro_generator
from scipy.stats import norm

from nmma.eos import eos_likelihood as el
from nmma.eos.eos_processing import EoSConverter


class EoSConstraintTestContainer:
    """Constraints evaluated on the curves of tabulated EOSs, held by a macro-only
    converter as during sampling. Subclasses define ``make_constraint``."""

    def setup_method(self) -> None:
        self.converter = EoSConverter(generator=eos_macro_generator(), macro_only=True)
        self.parameters = self.converter({"EOS": np.arange(N_EOS_MACRO)})
        self.constraint = self.make_constraint()
        el.JointEoSConstraint(self.constraint, eos_converter=self.converter)

    def test_log_likelihood_has_the_batch_shape(self) -> None:
        log_l = self.constraint.log_likelihood(self.parameters)
        assert np.shape(log_l) == (N_EOS_MACRO,)


class MassConstraintTestContainer(EoSConstraintTestContainer):
    """Gaussian bounds on the TOV mass; subclasses set the ``expected_lognorm``
    of their bound."""

    measured_mass = 2.2
    measure_error = 0.1

    def test_log_likelihood_of_the_tov_masses(self) -> None:
        log_l = self.constraint.log_likelihood(self.parameters)
        expected = self.expected_lognorm(
            self.parameters["TOV_mass"],
            loc=self.measured_mass,
            scale=self.measure_error,
        )
        assert log_l == pytest.approx(expected)


class TestLowerMTOVConstraint(MassConstraintTestContainer):
    expected_lognorm = staticmethod(norm.logcdf)

    def make_constraint(self) -> el.LowerMTOVConstraint:
        return el.LowerMTOVConstraint(self.measured_mass, self.measure_error)


class TestUpperMTOVConstraint(MassConstraintTestContainer):
    expected_lognorm = staticmethod(norm.logsf)

    def make_constraint(self) -> el.UpperMTOVConstraint:
        return el.UpperMTOVConstraint(self.measured_mass, self.measure_error)


class TestMassRadiusConstraint(EoSConstraintTestContainer):
    """A measurement drawn around R_14 of the ``measured_eos``."""

    measured_eos = 0
    n_samples = 20000
    measured_mass = 1.4
    mass_error = 0.05
    radius_error = 0.2
    # above the TOV masses of all test EOSs (~2.2), also after the 5% padding of
    # the histogram range, but within the test masses
    heavy_mass = 2.45
    heavy_mass_error = 0.01
    seed = 1

    def make_constraint(self) -> el.MassRadiusConstraint:
        self.rng = np.random.default_rng(self.seed)
        measured_radius = self.parameters["R_14"][self.measured_eos]
        return self.measurement(self.measured_mass, self.mass_error, measured_radius)

    def measurement(
        self, mass: float, mass_error: float, radius: float
    ) -> el.MassRadiusConstraint:
        masses = self.rng.normal(mass, mass_error, self.n_samples)
        radii = self.rng.normal(radius, self.radius_error, self.n_samples)
        return el.MassRadiusConstraint(masses, radii)

    def test_measured_eos_is_most_likely(self) -> None:
        log_l = self.constraint.log_likelihood(self.parameters)
        assert np.argmax(log_l) == self.measured_eos

    def test_masses_above_the_tov_mass_do_not_contribute(self) -> None:
        "There the curves have radius 0, outside every measurement"
        measured_radius = self.parameters["R_14"][self.measured_eos]
        heavy = self.measurement(
            self.heavy_mass, self.heavy_mass_error, measured_radius
        )
        el.JointEoSConstraint(heavy, eos_converter=self.converter)
        log_l = heavy.log_likelihood(self.parameters)
        assert np.all(log_l == -np.inf)
