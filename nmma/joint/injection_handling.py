import numpy as np
import pandas as pd
import os
import bilby
from bilby_pipe.utils import convert_string_to_dict
from bilby_pipe.create_injections import InjectionCreator


from ..core.parsing import (parsing_and_logging, slurm_setup_parser, nmma_base_parsing, process_multi_condition_string)
from ..core.constants import set_cosmology, get_cosmology
from ..core.utils import set_filename, rejection_sample, read_injection_file
from ..core.conversion import MultimessengerConversion, KilonovaEjectaFitting, bbh_source_frame
from ..em import utils, lightcurve_handling as lch
from ..em.model import create_injection_model
from ..eos.eos_processing import EoSConverter
from .joint_parsing import injection_parsing

class NMMAInjectionCreator(InjectionCreator):
    """A class to create NMMA injections, extending the bilby_pipe InjectionCreator."""

    def __init__(self, args, **kwargs):
        self.args = args
        # can use a prior_dict or a prior_file
        if isinstance(args.prior_dict, str):
            # convert string to dict
            args.prior_dict = convert_string_to_dict(args.prior_dict)

        set_cosmology(getattr(args, 'cosmology', None))
        super().__init__(
            prior_file=args.prior_file,
            prior_dict=args.prior_dict,
            n_injection=args.n_injection,
            default_prior="CBCPriorDict",
            trigger_time=getattr(args, "trigger_time", 0.),
            deltaT=args.deltaT,
            gpstimes=args.gps_file,
            duration=args.duration,
            post_trigger_duration=args.post_trigger_duration,
            generation_seed=args.generation_seed,
            cosmology=get_cosmology(),
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
             ## initialise testing methods we want to apply to the injections
            self.setup_test_routines(args)
            self.columns_to_remove = None
            self.max_redraws = args.max_redraws

            # initialise post-processing methods for the injections
            self.setup_post_processing(args)

            # we need to be able to do a parameter conversion
            self.param_conversion = MultimessengerConversion.from_dict(self.conv_instructions)

            self.original_parameters = args.original_parameters
            self.include_checks = True

        # legacy
        self.gw_injection_file = getattr(args, 'gw_injection_file', self.filename)
        self.reference_frequency = getattr(args, "reference_frequency", 20.0)

    def setup_test_routines(self, args):
        self.conv_instructions = {}
        test_methods = []
        tests = process_multi_condition_string(args.tests)

        for test, val in tests.items():
            if 'snr' in test:
                self.initialise_ifos(args)
                self.conv_instructions['gw'] = bbh_source_frame
                self.snr_op, self.snr_threshold = val
                test_methods.append(self.test_snr)
            elif 'population' in test:
                test_methods.append(self.test_population)
            elif 'ejecta' in test:
                test_methods.append(self.test_ejecta)
                self.conv_instructions['ejecta'] = True
            elif 'peak_magnitude' in test:
                self.mag_op, self.ref_mag = val
                self.initialise_lc_model(args)
                self.conv_instructions['em'] = self.lc_model.parameter_conversion
                test_methods.append(self.test_detectability)

        self.conv_instructions['eos'] = EoSConverter(args) 
        if "Hubble_constant" in self.priors:
            self.conv_instructions['cosmo'] = self.cosmology
        self.test_routines = test_methods
                   
    def setup_post_processing(self, args):
        postprocess_methods = []
        if 'snr' in args.post_processing and not hasattr(self, 'snr_threshold'):
            self.initialise_ifos(args)
            self.conv_instructions['gw'] = bbh_source_frame
            postprocess_methods.append(self.add_snrs)
        if 'ejecta' in args.post_processing:
            postprocess_methods.append(self.compute_ejecta)
        if 'lightcurve' in args.post_processing:
            self.initialise_lc_model(args)
            postprocess_methods.append(self.prepare_lightcurves)

        if not postprocess_methods:
            postprocess_methods.append(lambda df: df)  # No-op if no postprocessing is needed
        self.postprocessing = postprocess_methods
    
    def generate_injection_file(self):
        """Generate the injection file based on the provided parameters."""   
        dataframe = self.generate_prelim_dataframe()
        if self.include_checks:
            dataframe = self.testing_and_postprocessing(dataframe)

        # Finally dump the whole thing back into a json injection file
        self.write_injection_dataframe(dataframe, self.filename, self.extension )    

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

    def adjusted_prior_draw(self):
        dataframe_from_prior = self.get_injection_dataframe()
        try: ## FIXME: This could be handled more gracefully...
            swap_mask = dataframe_from_prior["mass_1"] < dataframe_from_prior["mass_2"]    
            dataframe_from_prior.loc[swap_mask, ['mass_1', 'mass_2']] = dataframe_from_prior.loc[swap_mask, ['mass_2','mass_1']].values
        except KeyError:
            pass
        if self.columns_to_remove is not None:
            dataframe_from_prior.drop(columns=self.columns_to_remove, inplace=True)
        return dataframe_from_prior

    def testing_and_postprocessing(self, dataframe):
        # step 3: redraw if necessary until all injections passed the tests
        test_df = self.test_wrap(dataframe)

        if test_df['tests_passed'].all():
            dataframe = test_df
        else:
            dataframe['tests_passed'] = test_df['tests_passed']
            dataframe = self.refill_failed_tests(dataframe)
        dataframe.drop(columns=['tests_passed'], inplace=True)
  
        # step 4: Wrap things up
        # do final conversion to all necessary parameters if desired
        if not self.original_parameters:
            dataframe = self.param_conversion.core_conversion(dataframe)
        # or add expensive information not required for tests
        for postprocess in self.postprocessing:
            postprocess(dataframe)
        return dataframe
    
    def test_wrap(self, dataframe):
        test_df = dataframe.copy()
        test_df = self.param_conversion.core_conversion(test_df)
        test_df['tests_passed'] =self.priors.evaluate_constraints(test_df)
        for test_routine in self.test_routines:
            # run the test routine on the test_df
            # this will modify the dataframe in place
            test_routine(test_df)
        
        return test_df
    
    def refill_failed_tests(self, dataframe):
        """Routine to redo tests until all conditions are fulfilled or max_redraws is reached."""
        redraw_from_prior = self.adjusted_prior_draw()
        redraws = 1
        while redraws <= self.max_redraws: 
            fail_mask = ~ dataframe['tests_passed'].astype(bool)
            n_fail = fail_mask.sum()
            if n_fail == 0:                 # if all tests passed, break
                return dataframe
            
            if len(redraw_from_prior) < n_fail:
                redraws += 1
                extra_draw = self.adjusted_prior_draw()
                redraw_from_prior = pd.concat([redraw_from_prior, extra_draw], ignore_index=True)

            replace_vals = redraw_from_prior.iloc[:n_fail][self.use_prior_columns]
            redraw_from_prior = redraw_from_prior.iloc[n_fail:]

            dataframe.loc[fail_mask, self.use_prior_columns]=replace_vals.to_numpy()
            retest_df = dataframe[fail_mask].reset_index(drop=True)
            retest_df['tests_passed'] = self.test_wrap(retest_df)
            dataframe.loc[fail_mask, 'tests_passed'] = retest_df['tests_passed'].to_numpy()

        raise ValueError(f"Redrew {redraws} times, but still {n_fail} failed samples. "
            "Consider increasing the max_redraws or check your prior." )

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

    def test_population(self, df):
        # FIXME: Allow tests on other distributions
        
        # rejection sampling for uniform mass ratio
        pop_prob= BNS_distribution(df["mass_1"], df["mass_2"])
        df["tests_passed"] *= rejection_sample(pop_prob, np.ones_like(pop_prob), self.rng)[1]
        # min mass constraint
        df["tests_passed"]*=(df["mass_2_source"] >=1.)

    def test_ejecta(self, df):
        df["tests_passed"]*=np.isfinite(df["log10_mej_dyn"]) 
        df["tests_passed"]*=np.isfinite(df["log10_mej_wind"])
    
    def test_snr(self, df):
        """Test the SNR of the injections."""
        df = self.add_snrs(df)
        df["tests_passed"]*= self.snr_op(df["snr"], self.snr_threshold)
    
    def test_detectability(self, df):
        """Test whether the injections are detectable in the light curve model."""
        # FIXME: Extend to respect known systems / filters
        def row_check(data_row):
            _, mags = self.lc_model.gen_detector_lc(data_row)
            return any([(self.mag_op(mag, self.ref_mag)).any() for mag in mags.values()])
        df["tests_passed"] *= df.apply(lambda row: row_check(row), axis=1)

    #################### Messenger-specific methods ########################
    def initialise_ifos(self, args):
        if isinstance(args.gw_detectors, str):  
            detectors = args.gw_detectors.split(",")
        else:
            detectors = args.gw_detectors
        dets = [bilby.gw.detector.get_empty_interferometer(det) for det in detectors 
                     if det != 'ET']
        if 'ET' in detectors:
            dets.append(bilby.gw.detector.networks.get_empty_interferometer('ET'))

        self.ifos = bilby.gw.detector.InterferometerList(dets)
        self.f_min = max(ifo.minimum_frequency for ifo in self.ifos)
        f_max = min(ifo.maximum_frequency for ifo in self.ifos)

        self.sampling_frequency = f_max * 2  

        default_waveform_arguments = {'reference_frequency': 20.0, 'waveform_approximant': 'IMRPhenomXAS_NRTidalv3', 'minimum_frequency': self.f_min, 'maximum_frequency': f_max}
        if args.waveform_arguments:
            waveform_arguments = convert_string_to_dict(args.waveform_arguments)
            waveform_arguments = default_waveform_arguments | waveform_arguments

        self.duration = 2048.

        self.waveform_gen = bilby.gw.WaveformGenerator(
            sampling_frequency=self.sampling_frequency, duration=self.duration,
            start_time=2-self.duration, time_domain_source_model= None,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments)

    def add_snrs(self, dataframe):
        """Compute SNR and duration for each row in parallel using threads."""
        # FIXME: preferable to parallelise, but ifo meta_data is not thread-safe
        
        # records = dataframe.to_dict("records")
        # n_workers = min(len(records), os.cpu_count() or 1)
        # with ThreadPool(n_workers) as pool:
        #     results = pool.map(self.compute_snr, records)

        
        # snrs, durations = zip(*results)
        # dataframe[["snr", "duration"]] = np.column_stack((snrs, durations))

        dataframe[["snr", "duration"]] = dataframe.apply(
            lambda row: self.compute_snr(row), axis=1, result_type='expand')
        return dataframe
    
    def compute_snr(self, data):
        injection_parameters = {'fiducial': 1. }
        injection_keys = ('mass_1', 'mass_2','chi_1', 'chi_2', 'luminosity_distance', 'dec', 'ra', 'theta_jn', 'phase', 'theta_jn', 'psi', 'geocent_time', 'lambda_1', 'lambda_2')
        for k in injection_keys:  
            injection_parameters[k] = data[k]

        self.ifos.set_strain_data_from_zero_noise(self.sampling_frequency, duration=self.duration, start_time=2-self.duration) 
        self.ifos.inject_signal(injection_parameters, waveform_generator=self.waveform_gen)
        

        mass_1,mass_2 = injection_parameters['mass_1'], injection_parameters['mass_2']
        chi_1,chi_2 = injection_parameters['chi_1'], injection_parameters['chi_2']
        chi_eff = (mass_1 * chi_1 + mass_2 * chi_2)/(mass_1+mass_2) 
        true_duration = np.rint(bilby.gw.utils.calculate_time_to_merger(self.f_min, mass_1, mass_2, chi=chi_eff))+1
        return np.sqrt(np.sum([ifo.meta_data['optimal_SNR']**2 for ifo in self.ifos]) ), true_duration
    
    
    def initialise_lc_model(self, args):
        self.lc_model = create_injection_model(args)
        self.args.label = args.lc_label if args.lc_label else self.filename
        self.detection_limit = utils.create_detection_limit(args, self.lc_model.filters)

    def prepare_lightcurves(self, dataframe):
        lch.create_multiple_injections(dataframe, self.args, self.lc_model, format= 'standard')

    def compute_ejecta(self, dataframe):
        """Compute the ejecta parameters for the injections."""
        # very short wrapper to avoid issues in the initialisation sequence
        return KilonovaEjectaFitting(dataframe)
            

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

def multi_run_setup():
    args = parsing_and_logging(slurm_setup_parser)
    injection_creator = NMMAInjectionCreator(args)
    dataframe = injection_creator.generate_prelim_dataframe()

    for index, _ in dataframe.iterrows():
        outdir = os.path.join(args.outdir, str(index))
        os.makedirs(outdir, exist_ok=True)
        injection_creator.priors.to_file(outdir, label="injection")
        with open(args.analysis_file, "r") as file:
            analysis = file.read()

        for key, data in zip(
            ('PRIOR', 'OUTDIR', 'INJOUT', 'INJNUM'), 
            (os.path.join(outdir, "injection.prior"), outdir, os.path.join(outdir, "lc.csv"), str(index))
        ):
            analysis = analysis.replace(key, data)

        with open(os.path.join(outdir, "inference.sh"), "w") as file:
            file.write(analysis)


        
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

def main(args=None):
    generate_injection(args)


# Use the routine by executing the script directly
if __name__ == "__main__":
    generate_injection()