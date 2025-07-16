import json
import os
from pathlib import Path
import yaml
from ast import literal_eval

import bilby
import matplotlib
import numpy as np
import pandas as pd
from bilby.core.likelihood import ZeroLikelihood

from matplotlib.pyplot import cm

from .lightcurve_handling import make_injection
from .em_likelihood import EMTransientLikelihood
from .prior import create_prior_from_args
from . import io, model, utils  
from .plotting_utils import basic_em_analysis_plot, bolometric_lc_plot
from .em_parsing import parsing_and_logging, multi_wavelength_analysis_parser, bolometric_parser
from ..joint.conversion import convert_mtot_mni
from ..joint.utils import read_injection_file, set_filename, fetch_bestfit
matplotlib.use("agg")

def data_from_injection(args, filters, detection_limit):
    injection_df = read_injection_file(args)
    injection_parameters = injection_df.iloc[args.injection_num].to_dict()
    data, injection_parameters = make_injection(injection_parameters, args, filters)
    data = inspect_detection_limit(detection_limit, data)
    inj_outfile = set_filename(args.label, args, f"_lc_{args.injection_num}")
    io.write_em_observations(inj_outfile, data, format='model')
    return data, injection_parameters

def inspect_detection_limit(detection_limit, data):
    #checking produced data for magnitudes dimmer than the detection limit
    for filt, limit in detection_limit.items():
        filt_dict = data[filt]
        non_detections = filt_dict['mag'] > limit

        filt_dict['mag'] = np.where(non_detections, limit, filt_dict['mag'])
        filt_dict['mag_error'] = np.where(non_detections, np.inf,filt_dict['mag_error'] )
    return data

def check_detections(data, remove_nondetections=False):
    if remove_nondetections:
        for filt, filt_dict in data.items():
            detections = np.isfinite(filt_dict['mag_error'])
            if detections.any():
                data[filt] = {k: v[detections] for k, v in filt_dict.items()}
            else:
                data.pop(filt)

    if not any(np.isfinite(data[filt]['mag_error']).any()  for filt in data):
        print("No detection available, fits only on non-detections.")
    return data

def set_analysis_filters(filters, data):
    if filters is None:
        filters = list(data.keys())

    filters_to_analyze = [filt for filt in data.keys() if filt in filters]
    print(f"Running with filters {filters_to_analyze}")
    return filters_to_analyze

def em_only_sampling(likelihood, priors, args):

    if args.bilby_zero_likelihood_mode:
        likelihood = ZeroLikelihood(likelihood)

    # fetch the additional sampler kwargs
    sampler_kwargs = literal_eval(args.sampler_kwargs)
    print("Running with the following additional sampler_kwargs:")
    print(sampler_kwargs)

    # check if it is running with reactive sampler
    if args.reactive_sampling:
        if args.sampler != "ultranest":
            print("Reactive sampling is only available in ultranest")
        else:
            print("Running with reactive-sampling in ultranest")
            nlive = None
    else:
        nlive = args.nlive

    if args.skip_sampling:
        print("Sampling for 1 iteration and plotting checkpointed results.")
        if args.sampler == "pymultinest":
            sampler_kwargs["max_iter"] = 1
        elif args.sampler == "ultranest":
            sampler_kwargs["niter"] = 1
        elif args.sampler == "dynesty":
            sampler_kwargs["maxiter"] = 1
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler=args.sampler,
        outdir=args.outdir,
        label=args.label,
        nlive=nlive,
        seed=args.sampling_seed,
        soft_init=args.soft_init,
        queue_size=args.cpus,
        check_point_delta_t=3600,
        **sampler_kwargs,
    )
    # check if it is running under mpi
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            pass
        else:
            return

    except ImportError:
        pass
    result.save_posterior_samples()
    return result

def post_process_bestfit(bestfit_params, transient, args, result=None):
    best_mags = bestfit_lightcurve(transient, bestfit_params)
    model_error = {filt: transient.compute_em_err(filt, best_mags["time"])
                    for filt in best_mags.keys() if filt != "time"}
    # model may not necessarily work on observed filters:
    for filt in set(transient.observed_filters) - set(best_mags.keys()):
        best_mags[filt] =  utils.get_filtered_mag(best_mags, filt)
        model_error[filt]= utils.get_filtered_mag(model_error, filt)

    
    transient.parameters = bestfit_params
    chi2_dict, mismatches = compute_chisquare_dict(transient, best_mags, 
                                        model_error, verbose=args.verbose)

    if getattr(args, "bestfit", False):
        bestfit_to_write = bestfit_params.copy()
        if result is not None:
            bestfit_to_write["log_bayes_factor"] = result.log_bayes_factor
            bestfit_to_write["log_bayes_factor_err"] = result.log_evidence_err
        bestfit_to_write["Magnitudes"] = {filt: best_mags[filt].tolist() 
                                          for filt in transient.observed_filters}
        bestfit_to_write["chi2_per_dof"] = chi2_dict["total"]
        bestfit_to_write["chi2_dict"] = chi2_dict
        bestfit_file = os.path.join(args.outdir, f"{args.label}_bestfit_params.json")

        with open(bestfit_file, "w") as file:
            json.dump(bestfit_to_write, file, indent=4)

        print(f"Saved bestfit parameters and magnitudes to {bestfit_file}")

    if args.plot:
        filters_to_plot = [
            filt for filt in transient.observed_filters
            if not np.isnan(transient.light_curves[filt]).all()
        ]
        plot_error = {filt: model_error[filt] for filt in filters_to_plot}
        mags_to_plot = {filt: best_mags[filt] for filt in filters_to_plot}
        mags_to_plot["time"] = best_mags["time"]

        if isinstance(transient.light_curve_model, model.CombinedLightCurveModelContainer):
            sub_models = transient.light_curve_model.models
            model_colors = cm.Spectral(np.linspace(0, 1, len(sub_models)))[::-1]
            obs_times , mag_all = transient.light_curve_model.gen_detector_lc(
                bestfit_params, return_all=True
            )
            sub_model_plot_props = {}
            for i, sub_model in enumerate(sub_models):
                sub_model_plot_props[sub_model.model] ={
                    'color': model_colors[i], 
                    'plot_mags' : [utils.get_filtered_mag(mag_all[i], filt) for filt in filters_to_plot],
                    'plot_times': obs_times[i]
                }
        else: sub_model_plot_props = None

        
        basic_em_analysis_plot(
            transient, mags_to_plot, plot_error, chi2_dict, mismatches,
            sub_model_plot_props, xlim = args.xlim, ylim = args.ylim, 
            save_path = os.path.join(args.outdir, f"{args.label}_lightcurves.png")
        )

def bestfit_lightcurve(transient, bestfit_params, sample_times=None):
    
    light_curve_model = transient.light_curve_model
    observable_times, obs_lightcurve = light_curve_model.gen_detector_lc(
        bestfit_params, sample_times
    )
    if not isinstance(obs_lightcurve, dict): # bolometric model, have to turn it into a dict
        obs_lightcurve = {'lbol': obs_lightcurve}
    obs_lightcurve["time"] = observable_times


    return obs_lightcurve
       
def compute_chisquare_dict(transient, model_data, model_error, verbose=False):
    chi2 = 0.0
    dof = 0.0
    chi2_dict = {}
    mismatches = {}
    for filt, mag  in transient.light_curves.items():
        t = transient.light_curve_times[filt]
        sigma_y = transient.light_curve_uncertainties[filt]
        # only the detection data are needed
        finite_idx = np.isfinite(sigma_y)
        n_finite = finite_idx.sum()
        if n_finite > 0:
            t_det, y_det, sigma_y_det = (
                t[finite_idx],
                mag[finite_idx],
                sigma_y[finite_idx],
            )

            offset = (y_det - np.interp(t_det,model_data["time"], model_data[filt])) ** 2
            total_unc = sigma_y_det**2 + model_error[filt]**2
            chi2_per_filt = np.sum(offset / total_unc)
            # store the data
            chi2 += chi2_per_filt
            dof += n_finite
            mismatches[filt] = (offset, total_unc)
            chi2_dict[filt] = float(chi2_per_filt / n_finite)

            if verbose:
                print(f"the {filt} data being analyzed is: ", t, y, sigma_y)
                print(f"for {filt} the length of the detections array is: ", n_finite, "increasing the dof to", dof)

    chi2_dict["total"] = chi2 / dof if dof > 0 else np.inf
    chi2_dict["dof"] = dof

    return chi2_dict, mismatches


def bolometric_analysis(args):

    # create the data 
    # FIXME add  injection functionality
    # if args.injection_file:
    #     pass

    # load the bolometric data
    data = pd.read_csv(args.light_curve_data)
    light_curve_model = model.SimpleBolometricLightCurveModel(
        model = args.em_model,
        sample_times = utils.setup_sample_times(args),  ## usually None, defaults to model_times
        )

    # setup the prior
    priors = create_prior_from_args(args)

    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        light_curve_data=data,
        priors=priors,
        error_budget=args.error_budget,
        verbose=args.verbose,
    )
    likelihood = EMTransientLikelihood(**likelihood_kwargs)
    result = em_only_sampling(likelihood, priors, args)

    result.plot_corner()

    if args.bestfit or args.plot:
        transient = likelihood.sub_model
        bestfit_params = fetch_bestfit(args)
        lbol_dict  = bestfit_lightcurve(transient, bestfit_params)

        bolometric_lc_plot(transient, lbol_dict,
            save_path = os.path.join(args.outdir, f"{args.label}_lightcurves.png")
        )
    return

def analysis(args):
    filters = utils.set_filters(args)
    detection_limit = utils.create_detection_limit(args, filters)

    # create the data if an injection set is given
    if args.injection_file:
        data, injection_parameters = data_from_injection(args, filters, detection_limit)
        trigger_time = injection_parameters['trigger_time']
        
    else:
        # load observational data
        data = io.load_em_observations(args, format='observations')

        trigger_time = utils.read_trigger_time(None,args)

    data = check_detections(data, args.remove_nondetections)
    filters_to_analyze = set_analysis_filters(filters, data)

    # initialize light curve model

    print("Creating light curve model for inference")
    lc_model_type = model.identify_model_type(args)
    light_curve_model = model.create_light_curve_model_from_args(
        lc_model_type, args, filters=filters_to_analyze,
    )
    try:
        model_names = [light_curve_model.model]
    except AttributeError:
        model_names = [sub_model.model for sub_model in light_curve_model.models]

    
    # setup the prior
    if any(model in ['AnBa2022_linear', 'AnBa2022_log'] for model in model_names):
        param_conv = convert_mtot_mni
    # elif to be extended...
    else:
        param_conv = None
    priors = create_prior_from_args(args, param_conv = param_conv)

    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters_to_analyze,
        light_curve_data=data,
        priors=priors,
        trigger_time=trigger_time,
        error_budget=args.em_error_budget,
        verbose=args.verbose,
        detection_limit=detection_limit,
        systematics_file=args.systematics_file
    )

    likelihood = EMTransientLikelihood(**likelihood_kwargs)

    result = em_only_sampling(likelihood, priors, args)
    
    if args.injection_file:
        injlist_all = ["luminosity_distance"]
        for model_name in model_names:
            add_params = model.model_parameters_dict[model_name]
            # FIXME: This seems very unnecessary.
            # But just in case there is a hidden purpose, let's keep it for this time.
            if "Bu2019" in model_name:
                try:
                    add_params.remove("KNtheta")
                    add_params.append("inclination_EM")
                except:
                    pass
            injlist_all += add_params
        ## A set with all the different parameters under analysis
        injlist_all = set(injlist_all)
        # A set of all parameters we do not vary in our analysis
        constant_columns = {col for col in result.posterior if len(result.posterior[col].unique()) == 1}
        var_inj_params = injlist_all - constant_columns
        injection = {key: injection_parameters[key] 
                        for key in var_inj_params 
                        if key in injection_parameters}
        result.plot_corner(parameters=injection)
    else:
        result.plot_corner()

    if args.bestfit or args.plot:
        bestfit_params = fetch_bestfit(args)
        post_process_bestfit(bestfit_params, likelihood.sub_model, args)

def nnanalysis(args):

    # import functions
    from ..mlmodel.dataprocessing import pad_the_data
    from ..mlmodel.embedding import SimilarityEmbedding
    from ..mlmodel.normalizingflows import normflow_params
    from ..mlmodel.inference import cast_as_bilby_result
    import torch
    from  nflows.flows import Flow

    # only continue if the Kasen model is selected
    if args.em_model != "Ka2017":
        print(
            "WARNING: model selected is not currently compatible with this inference method"
        )
        exit()

    # only can use ztfr, ztfg, and ztfi filters in the light curve data
    print('Currently filters are hardcoded to ztfr, ztfi, and ztfg. Continuing with these filters.')
    filters = 'ztfg,ztfi,ztfr'.split(",")

    detection_limit = utils.create_detection_limit(args, filters, 22.)

    # create the kilonova data if an injection set is given
    if args.injection_file:
        data, injection_parameters = data_from_injection(args, filters, detection_limit)
    else:
        # load the lightcurve data
        data = io.load_em_observations(args)
        

    data = check_detections(data, args.remove_nondetections)
    filters_to_analyze = set_analysis_filters(filters, data)
    lc_model_type = model.identify_model_type(args)
    light_curve_model = model.create_light_curve_model_from_args(
        lc_model_type, args, filters=filters_to_analyze,
    )
    
    # setup the prior
    priors = create_prior_from_args(args)
    
    # now that we have the kilonova light curve, we need to pad it with non-detections
    # this part is currently hard coded in terms of the times !!!! likely will need the most work
    # (so that the 'fixed' and 'shifted' are properly represented)
    num_points = 121
    num_channels = 3
    time_step = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Convert data dict to DataFrame with time and filter columns
    res = next(iter(data))
    t_list = (data[res]['time']).tolist()
    data_df = pd.DataFrame({'t': t_list})
    for key in data:
        data_df[key] = data[key]['mag']
    column_list = data_df.columns.to_list()

    # pad the data

    padded_data_df = pad_the_data(
        data_df, 
        column_list,
        desired_count=num_points, 
        filler_time_step=time_step, 
        filler_data=detection_limit[column_list[-1]], ## some value from the detection limit dict
    )
    # change the data into pytorch tensors
    data_tensor = torch.tensor(padded_data_df.iloc[:, 1:4].values.reshape(1, num_points, num_channels), dtype=torch.float32).transpose(1, 2)

    # set up the embedding 
    similarity_embedding = SimilarityEmbedding(num_dim=7, num_hidden_layers_f=1, num_hidden_layers_h=1, num_blocks=4, kernel_size=5, num_dim_final=5).to(device)
    num_dim = 7
    SAVEPATH = os.getcwd() + '/nmma/mlmodel/similarity_embedding_weights.pth'
    similarity_embedding.load_state_dict(torch.load(SAVEPATH, map_location=device))
    for name, param in similarity_embedding.named_parameters():
        param.requires_grad = False

    # set up the normalizing flows
    transform, base_dist, embedding_net = normflow_params(similarity_embedding, 9, 5, 90, context_features=num_dim, num_dim=num_dim) 
    flow = Flow(transform, base_dist, embedding_net).to(device=device)
    PATH_nflow = os.getcwd() + '/nmma/mlmodel/frozen-flow-weights.pth'
    flow.load_state_dict(torch.load(PATH_nflow, map_location=device))

    nsamples = 20000
    with torch.no_grad():
        samples = flow.sample(nsamples, context=data_tensor)
        samples = samples.cpu().reshape(nsamples,3)
        
    
    try:
        param_tensor = torch.tensor([injection_parameters['log10_mej'], injection_parameters['log10_vej'], injection_parameters['log10_Xlan']], dtype=torch.float32)
        with torch.no_grad():
            truth = param_tensor
    except NameError:
        truth = None
    except KeyError:
        raise ValueError('The injection parameters provided do not match the parameters the flow has been trained on')
    
    flow_result = cast_as_bilby_result(samples, truth, priors=priors)
    fig = flow_result.plot_corner(save=True, label = args.label, outdir=args.outdir)
    print('saved posterior plot')

def multi_analysis_loop(args, analysis_function):
    if getattr(args, 'config', None) is not None:
        yaml_dict = yaml.safe_load(Path(args.config).read_text())
        for analysis_set in yaml_dict.keys():
            params = yaml_dict[analysis_set]
            for key, value in params.items():
                key = key.replace("-", "_")
                if key not in args:
                    print(f"{key} not a known argument... please remove")
                    exit()
                setattr(args, key, value)
            analysis_function(args)
    else:
        analysis_function(args)

def main(args=None):
    args = parsing_and_logging(multi_wavelength_analysis_parser, args)
    if args.sampler == "neuralnet":
        analysis_function = nnanalysis
    else:
        analysis_function = analysis
    multi_analysis_loop(args, analysis_function)
    

def lbol_main(args=None):
    args = parsing_and_logging(bolometric_parser, args)
    multi_analysis_loop(args, bolometric_analysis)
