# FIXME: This is a hacky subclass to adapt the bilby_pipe data generation to 
# our needs. We should at some point get rid of bilby pipe

from bilby_pipe.data_generation import DataGenerationInput as DataInput
class NMMAGravitationalWaveInput(DataInput):
    def __init__(self, args, unknown_args):
        args.calibration_correction_type = 'data'
        super().__init__(args, unknown_args)
        self.interferometers.plot_data(outdir=self.data_directory, label=self.label)
        if self.likelihood_type == "ROQGravitationalWaveTransient":
            self.save_roq_weights()
        
        args.gw_likelihood_type = self.likelihood_type
    'Quick wrapper to fix some issues with the bilby_pipe data generation'

    @DataInput.interferometers.setter
    def interferometers(self, interferometers):
        self._detectors = [ifo.name for ifo in interferometers]
        self.minimum_frequency_dict = self.reset_frequency_dict(
            self.minimum_frequency_dict)
        self.maximum_frequency_dict = self.reset_frequency_dict(
            self.maximum_frequency_dict)
        DataInput.interferometers.fset(self, interferometers)

    def reset_frequency_dict(self, frequency_dict):
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