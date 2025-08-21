import json
import numpy as np
import pandas as pd

import bilby
from bilby_pipe.utils import convert_string_to_dict
from bilby_pipe.create_injections import InjectionCreator
import multiprocessing


from .base_parsing import nmma_base_parsing, process_multi_condition_string
from .joint_parsing import injection_parsing
from .utils import set_filename, rejection_sample, read_injection_file
from .conversion import MultimessengerConversion
from ..em import utils, lightcurve_handling as lch
from ..em.model import create_injection_model

class NMMAInjectionCreator(InjectionCreator):
    """A class to create NMMA injections, extending the bilby_pipe InjectionCreator."""

    def __init__(self, args, **kwargs):
        self.args = args
        # can use a prior_dict or a prior_file
        if isinstance(args.prior_dict, str):
            # convert string to dict
            args.prior_dict = convert_string_to_dict(args.prior_dict)
        super().__init__(
            prior_file=args.prior_file,
            prior_dict=args.prior_dict,
            n_injection=args.n_injection,
            default_prior="PriorDict",
            trigger_time=getattr(args, "trigger_time", 0.),
            deltaT=args.deltaT,
            gpstimes=args.gps_file,
            duration=args.duration,
            post_trigger_duration=args.post_trigger_duration,
            generation_seed=args.generation_seed
        )
        self.rng = np.random.default_rng(self.generation_seed)
        for key, value in kwargs.items():
            setattr(self, key, value)

        #bilby-pipe injection_creator can only handle dat or json
        self.extension = 'dat' if args.extension == 'csv' else args.extension
        self.filename= set_filename(args.injection_file, args)

        if args.simple_setup:
            self.include_checks = False
        else:
            self.include_checks = True

            self.messengers = set()
            self.modifiers = set()
            ## initialise testing methods we want to apply to the injections
            self.setup_test_routines(args)
            self.fail_mask = None
            self.columns_to_remove = None
            self.max_redraws = args.max_redraws

            # initialise post-processing methods for the injections
            self.setup_post_processing(args)

            # we need to be able to do a parameter conversion
            messengers, modifiers = self.determine_conversion_from_args(args)
            self.param_conversion = MultimessengerConversion(args, messengers, modifiers)

            self.original_parameters = args.original_parameters

        # legacy
        self.gw_injection_file = getattr(args, 'gw_injection_file', self.filename)
        self.reference_frequency = getattr(args, "reference_frequency", 20.0)

    def setup_post_processing(self, args):
        postprocess_methods = []
        if 'snr' in args.post_processing:
            self.initialise_ifos(args)
            postprocess_methods.append(self.add_snrs)
        if 'ejecta' in args.post_processing:
            postprocess_methods.append(self.compute_ejecta)
        if 'lightcurve' in args.post_processing:
            self.initialise_lc_model(args)
            postprocess_methods.append(self.prepare_lightcurves)

        if not postprocess_methods:
            postprocess_methods.append(lambda df: df)  # No-op if no postprocessing is needed
        self.postprocessing = postprocess_methods

    def setup_test_routines(self, args):
        test_methods = [self.test_initialisation]
        tests = process_multi_condition_string(args.tests)
        for test, val in tests.items():
            if 'snr' in test:
                self.snr_threshold = val[1]
                test_methods.append(self.test_snr)
            elif 'population' in test:
                test_methods.append(self.test_population)
            elif 'ejecta' in test:
                test_methods.append(self.test_ejecta)
            elif 'peak_magnitude' in test:
                self.ref_mag = val[1]
                self.initialise_lc_model(args)
                test_methods.append(self.test_detectability)

        self.test_routines = test_methods
                   
        

    def determine_conversion_from_args(self, args):
        """Determine the messengers and modifiers for the conversion based on the args."""
        # FIXME: To be extended
        messengers = []
        modifiers = []
        if 'ejecta' in (args.tests or args.post_processing):
            messengers.append('em')
        # eos conversion:
        if args.eos_file:
            modifiers.append('tabulated_eos')
            args.eos_to_ram = True
        elif args.micro_eos_model:
            messengers.append('eos') 
        return messengers, modifiers

    def generate_prelim_dataframe(self):
        #step 1: Check: we may want to extend a preliminary injection file
        # if not, this will be an empty dataframe
        dataframe_from_file = self.handle_incomplete_injection_file(self.gw_injection_file)
        if len(dataframe_from_file) > 0:
            self.n_injection = len(dataframe_from_file)

        # If we want to only complement a given injection, there might be an 
        # overlap in parameters. In that case, we drop the newly sampled ones
        prior_columns = set(self.priors.keys())
        inj_columns = set(dataframe_from_file.columns.tolist())
        self.columns_to_remove = list(inj_columns.intersection(prior_columns))

        # step 2: Get a first potential draw from the prior file
        # drop columns and adjust for mass ordering (m1 >= m2)
        dataframe_from_prior = self.adjusted_prior_draw()
        self.use_prior_columns = dataframe_from_prior.columns.tolist()
        # if 'index' in use_prior_columns:
        #     use_prior_columns.remove('index')
        
        
        # combine the dataframes
        dataframe = pd.DataFrame.merge(dataframe_from_file, dataframe_from_prior,
            how="outer", ## outer to keep all prior parameters
            left_index=True, right_index=True )
        
        # Move dataframe index column to simulation_id if column does not exist
        if "simulation_id" not in dataframe.columns:
            dataframe = dataframe.reset_index().rename({"index": "simulation_id"}, axis=1)
        dataframe.astype({'simulation_id': 'int32'})
        return dataframe
    
    def testing_and_postprocessing(self, dataframe):
        # step 3: Do a test and redraw if necessary
        dataframe = self.test_wrap(dataframe)

        # step 4: redo until sufficient injections have passed the tests
        dataframe = self.refill_failed_tests(dataframe)
        dataframe.drop(columns=['tests_passed'], inplace=True)
  
        # step 5: Wrap things up
        # do final conversion to all necessary parameters
        dataframe, added_keys = self.param_conversion.convert_to_multimessenger_parameters(dataframe)
        # remove the parameters that were added by the conversion if not desired
        if self.original_parameters:
            dataframe.drop(columns=added_keys, inplace=True)
        # or add expensive information not required for tests
        for postprocess in self.postprocessing:
            dataframe = postprocess(dataframe)
        return dataframe


    def generate_injection_file(self):
        """Generate the injection file based on the provided parameters."""   
        dataframe = self.generate_prelim_dataframe()
        if self.include_checks:
            dataframe = self.testing_and_postprocessing(dataframe)

        # Finally dump the whole thing back into a json injection file
        self.write_injection_dataframe(dataframe, self.filename, self.extension )


    def adjusted_prior_draw(self):
        dataframe_from_prior = self.get_injection_dataframe()
        try: ## FIXME: This could be handled more gracefully...
            swap_mask = dataframe_from_prior["mass_1"] < dataframe_from_prior["mass_2"]    
            dataframe_from_prior.loc[swap_mask, ['mass_1', 'mass_2']] = dataframe_from_prior.loc[swap_mask, ['mass_2','mass_1', ]].values
        except KeyError:
            pass
        if self.columns_to_remove is not None:
            dataframe_from_prior.drop(columns=self.columns_to_remove, inplace=True)
        return dataframe_from_prior


    def test_wrap(self, dataframe):
        ## test_df must be a copy since some tests require to modify the dataframe
        if self.fail_mask is None:
            test_df = dataframe.copy()
        else:
            test_df = dataframe[self.fail_mask].copy()

        # transform the parameters to multimessenger parameters and test them
        test_df, _ = self.param_conversion.convert_to_multimessenger_parameters(test_df)
        for test_routine in self.test_routines:
            # run the test routine on the test_df
            # this will modify the dataframe in place
            test_df = test_routine(test_df)
        
        dataframe.loc[test_df.index, 'tests_passed'] = test_df['tests_passed']
        self.fail_mask =~dataframe['tests_passed']
        self.n_fail = self.fail_mask.sum()
        return dataframe
    
    def refill_failed_tests(self, dataframe):
        redraws = 0
        while True: 
            if self.n_fail == 0:                 # if all tests passed, break
                break
            # replace the failed samples with new samples from the prior
            try:
                assert len(redraw_from_prior) >= self.n_fail
            except:
                # unless we first need to get new samples from the prior 
                # because the number of failed samples is larger 
                # than the number of unused new samples
                redraw_from_prior = self.adjusted_prior_draw()
                redraws += 1
                if redraws > self.max_redraws:  # try redraw only up to max_redraws times
                    raise ValueError(f"Redrew {redraws} times, but still {self.n_fail} failed samples. "
                        "Consider increasing the max_redraws or check your prior." )

            replace_vals = redraw_from_prior.iloc[:self.n_fail][self.use_prior_columns].reset_index(drop=True)

            dataframe.loc[self.fail_mask, self.use_prior_columns]=replace_vals.values
            redraw_from_prior = redraw_from_prior.iloc[self.n_fail:]

            dataframe = self.test_wrap(dataframe)
        return dataframe

    def handle_incomplete_injection_file(self, gw_injection_file=None):
        # check injection file format
        if gw_injection_file:
            if not gw_injection_file.endswith((".json", ".xml", ".xml.gz", ".dat")):
                raise ValueError("Unknown injection file format")

            # load the injection json file
            if gw_injection_file.endswith(".json"):
                dataframe_from_file = read_injection_file(gw_injection_file)
            else:
                # legacy formats preferably not used anymore
                # complement given gw-injection in nmma-style
                dataframe_from_file = self.file_to_dataframe(
                    gw_injection_file,
                    self.reference_frequency,
                    trigger_time=self.trigger_time,
                )
        else:
            dataframe_from_file = pd.DataFrame()
            print(
                "No injection files provided, "
                "will generate injection based on the prior file provided only"
            )

        return dataframe_from_file


    def test_initialisation(self, df):
        df["tests_passed"]=True

    def test_population(self, df):
        # FIXME: Allow tests on other distributions
        
        # rejection sampling for uniform mass ratio
        pop_prob= BNS_distribution(df["mass_1"], df["mass_2"])
        df["tests_passed"] *= rejection_sample(pop_prob, np.ones_like(pop_prob), self.rng)
        # min mass constraint
        df["tests_passed"]*=(df["mass_2_source"] >=1.)
        return df

    def test_ejecta(self, df):
        df["tests_passed"]*=np.isfinite(df["log10_mej_dyn"]) 
        df["tests_passed"]*=np.isfinite(df["log10_mej_wind"])
        return df
    
    def test_snr(self, df):
        """Test the SNR of the injections."""
        df = self.add_snrs(df)
        df["tests_passed"]*=(df["snr"] >= self.snr_threshold)
        return df
    
    def test_detectability(self, df):
        for data_row in df.itertuples():
            times, mags = self.lc_model.gen_detector_lc(data_row)
            if not any([(mag > self.ref_mag).any() for mag in mags]):
                df.at[data_row.Index, "tests_passed"] = False

    #################### Messenger-specific methods ########################
    def initialise_ifos(self, args):
        detectors = args.gw_detectors.split(",")
        dets = [bilby.gw.detector.get_empty_interferometer(det) for det in detectors 
                     if det != 'ET']
        if 'ET' in detectors:
            dets.append(bilby.gw.detector.networks.get_empty_interferometer('ET'))

        self.ifos = bilby.gw.detector.InterferometerList(dets)  # et_det,
        self.f_min = max(ifo.minimum_frequency for ifo in self.ifos)
        f_max = min(ifo.maximum_frequency for ifo in self.ifos)

        self.sampling_frequency = f_max * 2  

        default_waveform_arguments = {'reference_frequency': 20.0, 'waveform_approximant': 'IMRPhenomXAS_NRTidalv3', 'minimum_frequency': self.f_min, 'maximum_frequency': f_max}
        if args.waveform_arguments:
            waveform_arguments = convert_string_to_dict(args.waveform_arguments)
            waveform_arguments = default_waveform_arguments | waveform_arguments

        self.duration = 2048.

        self.waveform_generator = bilby.gw.WaveformGenerator(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=2-self.duration, time_domain_source_model= None,
            frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments)

    def add_snrs(self, dataframe):
        snr_list = []
        duration_list = []
        def worker(idx):
            return self.compute_snr(dataframe, idx)

        with multiprocessing.Pool() as pool:
            results = pool.map(worker, range(len(dataframe.index)))

        for snr, duration in results:
            snr_list.append(snr)
            duration_list.append(duration)
        dataframe["snr"] = snr_list
        dataframe["duration"] = duration_list
        return dataframe
    
    def compute_snr(self, dataframe, idx):
        injection_parameters = {'fiducial': 1. }
        injection_keys = ('mass_1', 'mass_2','chi_1', 'chi_2', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'phase', 'theta_jn', 'psi', 'geocent_time', 'lambda_1', 'lambda_2')
        for k in injection_keys:  
            injection_parameters[k] = dataframe[k][idx]

        self.ifos.set_strain_data_from_zero_noise(self.sampling_frequency, duration=self.duration, start_time=2-self.duration) 
        self.ifos.inject_signal(injection_parameters, waveform_generator=self.waveform_generator)
        

        mass_1,mass_2 = injection_parameters['mass_1'], injection_parameters['mass_2']
        chi_1,chi_2 = injection_parameters['chi_1'], injection_parameters['chi_2']
        chi_eff = (mass_1 * chi_1 + mass_2 * chi_2)/(mass_1+mass_2) 
        true_duration = np.rint(bilby.gw.utils.calculate_time_to_merger(self.f_min, mass_1, mass_2, chi=chi_eff))+1
        return np.sqrt(np.sum([ifo.meta_data['optimal_SNR']**2 for ifo in self.ifos]) ), true_duration
    
    
    def initialise_lc_model(self, args):
        self.lc_model = create_injection_model(args)
        self.label = args.lc_label if args.lc_label else self.filename
        self.detection_limit = utils.create_detection_limit(args, self.lc_model.filters)

    def prepare_lightcurves(self, dataframe):
        lch.create_multiple_injections(dataframe, self.args, self.lc_model, format= 'standard')

    def compute_ejecta(self, dataframe):
        """Compute the ejecta parameters for the injections."""
        # very short wrapper to avoid issues in the initialisation sequence
        return self.param_conversion.ejecta_parameter_conversion(dataframe)
            

    ############################ Legacy functions ##########################
    def file_to_dataframe(self,injection_file, reference_frequency, trigger_time=0.0
    ):
        """legacy function to convert a bilby- injection file to a dataframe.
        Consider doing a complete nmma-injection instead"""
        #legacy imports
        from  lalsimulation import SimInspiralTransformPrecessingWvf2PE as lalsim_conversion
        from gwpy.table import Table

        try:
            import ligo.lw  # noqa F401
        except ImportError:
            raise ImportError("You do not have ligo.lw installed: $ pip install python-ligo-lw")

        if injection_file.endswith((".xml", '.xml.gz')):
            table = Table.read(injection_file, format="ligolw", tablename="sim_inspiral")
        elif injection_file.endswith(".dat"):
            table = Table.read(injection_file, format="csv", delimiter="\t")
        else:
            raise ValueError("Only understand xml and dat")

        injection_values = {key: [] for key in ["simulation_id", "mass_1", "mass_2", 
            "luminosity_distance", "psi", "phase", "geocent_time", "ra", "dec", 
            "theta_jn", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl" ]}
        for row in table:
            coa_phase = row.get("coa_phase", 0)
            spin_args = [row.get("spin1x", 0.0), row.get("spin1y", 0.0), row["spin1z"],
                         row.get("spin2x", 0.0), row.get("spin2y", 0.0), row["spin2z"]]

            precession_args = [row["inclination"], *spin_args,
                row["mass1"], row["mass2"], reference_frequency, coa_phase]
            conversion_args = [float(arg) for arg in precession_args]
            conversion_keys = ["theta_jn","phi_jl", "tilt_1" ,"tilt_2", "phi_12", "a_1","a_2"]
            
            for key, val in zip(conversion_keys, lalsim_conversion(*conversion_args) ):
                injection_values[key].append(val)
    
            injection_values["simulation_id"].append(int(row["simulation_id"]))
            injection_values["luminosity_distance"].append(float(row["distance"]))
            injection_values["psi"].append(float(row.get("polarization", 0)))
            injection_values["ra"].append(float(row["longitude"]))
            injection_values["dec"].append(float(row["latitude"]))

            # ensure mass1 > mass2
            mass2, mass1 = np.sort([float(row["mass1"]), float(row["mass2"])])
            injection_values["mass_1"].append(float(mass1))
            injection_values["mass_2"].append(float(mass2))

            injection_values["phase"].append(float(coa_phase))
            geocent_time = float(row.get("geocent_end_time", trigger_time))
            geocent_time_ns = float(row.get("geocent_end_time_ns", 0)) * 1e-9
            injection_values["geocent_time"].append(geocent_time + geocent_time_ns)

        injection_values = pd.DataFrame.from_dict(injection_values)
        return injection_values

        
#### UTILS ######

def BNS_distribution(m1, m2):
    q= m2/m1
    return np.where(q<=1., q, 1/q)









############################ MAIN ###############################

def generate_injection(args = None):
    # step 0: parse the arguments
    # handle parsing similar to bilby-pipe and parse from config file
    if args is None:
        args = nmma_base_parsing(injection_parsing)
    
    injection_creator = NMMAInjectionCreator(args)
    injection_creator.generate_injection_file()


# Use the routine by executing the script directly
if __name__ == "__main__":
    generate_injection()