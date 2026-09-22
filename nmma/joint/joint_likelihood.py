import numpy as np
from bilby.core.likelihood import JointLikelihood

from ..core.base import NMMALikelihood, NMMALikelihoodMixin
from ..core.conversion import MultimessengerConversion
from ..em.em_likelihood import EMTransientLikelihood, setup_em_kwargs
from ..eos.eos_likelihood import (
    EoSConverter,
    EquationofStateLikelihood,
    setup_eos_kwargs,
)
from ..gw.gw_likelihood import GravitationalWaveTransientLikelihood, setup_gw_kwargs
from ..population.pop_likelihood import NeutronStarPopulation


class MultiMessengerLikelihood(NMMALikelihoodMixin, JointLikelihood):
    """
    A likelihood combining several messenger likelihoods.

    Parameters
    ----------
    messenger_likelihoods : list
        Likelihoods passed to ``bilby.core.likelihood.JointLikelihood``.
    priors : bilby.core.prior.PriorDict
        Assigned to ``self.priors``.
    conversion_instructions : dict, optional
        Assigned to ``self.conversion_instructions``, which
        :meth:`setup_parameter_conversion` reads.
    """

    def __init__(self, messenger_likelihoods, priors, conversion_instructions={}):
        super().__init__(*messenger_likelihoods)

        self.priors = priors
        self.conversion_instructions = conversion_instructions
        self.setup_parameter_conversion()
        self._noise_logl = JointLikelihood.noise_log_likelihood(self)

    def __repr__(self):
        """Name the class and the sub-likelihoods it holds."""
        if len(self.likelihoods) == 1:
            return f"{self.__class__.__name__} with {self.likelihoods[0].__repr__()}"
        else:
            reprs = [lhood.__repr__() for lhood in self.likelihoods]
            return f"{self.__class__.__name__} with {', '.join(map(str, reprs[:-1]))} and {reprs[-1]}"

    def setup_parameter_conversion(self):
        """
        Set ``multi_conversion`` from ``conversion_instructions``.

        The ``parameter_conversion`` of each EM, GW and EOS sub-likelihood is
        first recorded in ``conversion_instructions`` under its messenger key.
        If ``conversion_instructions`` is None, :meth:`parameter_conversion`
        and :meth:`posterior_conversion` are replaced by their ``basic_``
        counterparts and nothing else is done.

        Must be called after ``self.likelihoods`` is populated.
        """
        if self.conversion_instructions is None:
            self.parameter_conversion = self.basic_parameter_conversion
            self.posterior_conversion = self.basic_posterior_conversion
            return
        for lhood in self.likelihoods:
            if isinstance(lhood, EMTransientLikelihood):
                self.conversion_instructions["em"] = lhood.parameter_conversion
            elif isinstance(lhood, GravitationalWaveTransientLikelihood):
                self.conversion_instructions["gw"] = lhood.parameter_conversion
            elif isinstance(lhood, EquationofStateLikelihood):
                self.conversion_instructions["eos"] = lhood.parameter_conversion

        self.multi_conversion = MultimessengerConversion.from_dict(
            self.conversion_instructions
        )

    def sanity_checks(self):
        """
        Product of ``sanity_checks`` over the sub-likelihoods.

        Returns
        -------
        The value returned by ``numpy.prod``.
        """
        return np.prod([lhood.sanity_checks() for lhood in self.likelihoods])

    def sub_log_likelihood(self, parameters):
        """
        Sum ``sub_log_likelihood`` over the sub-likelihoods.

        Parameters
        ----------
        parameters
            Passed to each sub-likelihood.

        Returns
        -------
        The sum, or ``numpy.nan_to_num(-numpy.inf)`` if it is not finite.
        """
        logl = sum([lhood.sub_log_likelihood(parameters) for lhood in self.likelihoods])
        if np.isfinite(logl):
            return logl
        else:
            return np.nan_to_num(-np.inf)

    def final_diagnostics(self, bestfit_params, args, result=None):
        """
        Call ``final_diagnostics`` on each sub-likelihood.

        Parameters
        ----------
        bestfit_params
            Passed to each sub-likelihood.
        args
            Passed to each sub-likelihood.
        result : default=None
            Passed to each sub-likelihood.

        Returns
        -------
        list
            One entry per sub-likelihood.
        """
        return [
            lhood.final_diagnostics(bestfit_params, args, result)
            for lhood in self.likelihoods
        ]

    def parameter_conversion(self, samples):
        """
        Convert ``samples`` with ``self.multi_conversion``.

        Parameters
        ----------
        samples
            Passed to
            ``MultimessengerConversion.convert_to_multimessenger_parameters``.

        Returns
        -------
        That method's return value.
        """
        return self.multi_conversion.convert_to_multimessenger_parameters(samples)

    def basic_parameter_conversion(self, parameters):
        """
        Apply each sub-likelihood's ``parameter_conversion`` in turn.

        Parameters
        ----------
        parameters
            Passed to the first sub-likelihood; each result feeds the next.

        Returns
        -------
        The output of the last sub-likelihood.
        """
        for lhood in self.likelihoods:
            parameters = lhood.parameter_conversion(parameters)
        return parameters

    def posterior_conversion(self, posterior_samples):
        """
        Apply ``multi_conversion.core_conversion``, then
        :meth:`basic_posterior_conversion`.

        Parameters
        ----------
        posterior_samples : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        posterior = self.multi_conversion.core_conversion(posterior_samples)
        return self.basic_posterior_conversion(posterior)

    def basic_posterior_conversion(self, posterior_samples):
        """
        Apply each sub-likelihood's ``posterior_conversion`` in turn.

        Parameters
        ----------
        posterior_samples : pandas.DataFrame
            Passed to the first sub-likelihood; each result feeds the next.

        Returns
        -------
        pandas.DataFrame
            The result, restricted to numeric columns by ``select_dtypes``.
        """
        for lhood in self.likelihoods:
            posterior_samples = lhood.posterior_conversion(posterior_samples)
        return posterior_samples.select_dtypes([np.number])

    @classmethod
    def setup_from_args(cls, data_dump, priors, args, logger=None):
        """
        Build the likelihood for the messengers named in ``data_dump``.

        Parameters
        ----------
        data_dump : dict
            Read for ``messengers`` and ``analysis_modifiers``, and passed to
            each ``setup_*_kwargs``.
        priors : bilby.core.prior.PriorDict
            Passed to the messenger likelihoods.
        args : argparse.Namespace
            The parsed arguments.
        logger : logging.Logger, default=None
            If None, ``nmma.core.utils.logger`` is used.

        Returns
        -------
        The single messenger likelihood when only one was set up, otherwise a
        :class:`MultiMessengerLikelihood` over all of them.

        Raises
        ------
        ValueError
            If no messenger likelihood was set up.
        """
        if logger is None:
            from nmma.core.utils import logger

            logger = logger

        messengers = data_dump["messengers"]
        analysis_modifiers = data_dump["analysis_modifiers"]

        messenger_lhoods = []
        conversion_instructions = {}
        if "Hubble" in analysis_modifiers:
            conversion_instructions["cosmo"] = getattr(args, "cosmology", None)

        if "gw" in messengers:
            logger.info("Setting up GW likelihood")
            gw_kwargs = setup_gw_kwargs(data_dump, args, logger)
            messenger_lhoods.append(
                GravitationalWaveTransientLikelihood(priors, **gw_kwargs)
            )
            priors.convert_floats_to_delta_functions()
            conversion_instructions["gw"] = True  # placeholder

        if "eos" in messengers:
            logger.info("Sampling over EOS generated on the fly")
            eos_kwargs = setup_eos_kwargs(data_dump, args, logger)
            if eos_kwargs["constraint_dict"]:
                # only evaluate if specific constraints are given
                messenger_lhoods.append(EquationofStateLikelihood(priors, **eos_kwargs))
                conversion_instructions["eos"] = True  # placeholder
            else:
                conversion_instructions["eos"] = eos_kwargs["eos_converter"]
        elif "tabulated_eos" in analysis_modifiers:
            logger.info("Using tabulated EoS")
            conversion_instructions["eos"] = EoSConverter(args, "tabulated")
        elif "lambda_1" in priors and "lambda_2" in priors:
            logger.info("Using universal relations for tidal deformabilities")
            conversion_instructions["eos"] = EoSConverter(args, "qur")

        if "eos" in conversion_instructions and "gw" not in conversion_instructions:
            eos_converter = conversion_instructions["eos"]
            eos_converter.parameter_conversion = eos_converter.compute_macro_parameters
            conversion_instructions["eos"] = eos_converter

        if "em" in messengers:
            logger.info("Setting up EM likelihood")
            em_kwargs = setup_em_kwargs(priors, data_dump, args, logger)
            messenger_lhoods.append(EMTransientLikelihood(**em_kwargs))
            conversion_instructions["em"] = True  # placeholder

        if "pop" in messengers:
            pop_model = NeutronStarPopulation(args.population_model)
            messenger_lhoods.append(NMMALikelihood(pop_model, priors))
            # conversion_instructions['pop'] = 'model'

        # FIXME: Find a better way to check this automatically:
        # NOTE: this is nasty because we might have corresponding constraints, but do not need to...
        if (
            "log10_mej_wind" in priors
            or "log10_mej_dyn" in priors
            or args.ejecta_conversion
        ):
            conversion_instructions["ejecta"] = True

        if len(messenger_lhoods) == 0:
            raise ValueError("No messenger likelihoods were set up.")
        elif len(messenger_lhoods) == 1:
            lhood = messenger_lhoods[0]
            if "eos" in conversion_instructions:
                lhood.conv_functions.append(conversion_instructions["eos"])
            lhood.setup_parameter_conversion()
            logger.info(f"Only {lhood} is used. Using this instead.")
        else:
            lhood = cls(messenger_lhoods, priors, conversion_instructions)
            logger.info(f"Created {lhood}")
        return lhood
