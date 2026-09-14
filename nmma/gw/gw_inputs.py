# FIXME: This is a hacky subclass to adapt the bilby_pipe data generation to 
# our needs. We should at some point get rid of bilby pipe

from bilby_pipe.data_generation import DataGenerationInput as DataInput
class NMMAGravitationalWaveInput(DataInput):
    'Quick wrapper to fix some issues with the bilby_pipe data generation'

    def __init__(self, args, unknown_args):
        """Run bilby_pipe GW data generation with NMMA-specific fixups.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed generation config (as built by nmma-generation). Mutated
            in place: `calibration_correction_type` is forced to `'data'`
            before generation, and `gw_likelihood_type` is set afterwards
            to the resolved `self.likelihood_type`.
        unknown_args : list of str
            Leftover CLI arguments not recognized by the parser, forwarded
            to `DataGenerationInput`.
        """


        args.calibration_correction_type = 'data'
        super().__init__(args, unknown_args)
        self.interferometers.plot_data(outdir=self.data_directory, label=self.label)
        if self.likelihood_type == "ROQGravitationalWaveTransient":
            self.save_roq_weights()
        
        args.gw_likelihood_type = self.likelihood_type

    @DataInput.interferometers.setter
    def interferometers(self, interferometers):
        """Set interferometers, restricting frequency dicts to active detectors.

        Parameters
        ----------
        interferometers : bilby.gw.detector.InterferometerList
            Interferometers loaded for this run.

        Returns
        -------
        None
            Assigns `self._detectors` and rebuilt
            `minimum_frequency_dict`/`maximum_frequency_dict`, then
            delegates to the parent setter for the actual assignment.
        """

        self._detectors = [ifo.name for ifo in interferometers]
        self.minimum_frequency_dict = self.reset_frequency_dict(
            self.minimum_frequency_dict)
        self.maximum_frequency_dict = self.reset_frequency_dict(
            self.maximum_frequency_dict)
        DataInput.interferometers.fset(self, interferometers)

    def reset_frequency_dict(self, frequency_dict):
        """Filter a per-detector frequency dict down to `self.detectors`.

        Falls back to a shared prefix entry when a detector isn't found
        directly, e.g. `ET1`/`ET2`/`ET3` sharing a single `ET` entry for
        Einstein Telescope's sub-detectors.

        Parameters
        ----------
        frequency_dict : dict of {str : float}
            Minimum or maximum frequency [Hz] keyed by detector name.

        Returns
        -------
        dict of {str : float}
            Same units, restricted to keys in `self.detectors`.

        Raises
        ------
        ValueError
            If a detector in `self.detectors` has no matching entry,
            directly or via its prefix.

        """

        out_dict = {}
        for det in self.detectors:
            if det in frequency_dict:
                out_dict[det] = frequency_dict[det]
            # eg. ET1, ET2, ET3 were given by ET
            elif det[:-1] in frequency_dict:
                out_dict[det] = frequency_dict[det[:-1]]
            else:
                raise ValueError(
                    f"Detector {det} not found in frequency dict {frequency_dict}")
        return out_dict