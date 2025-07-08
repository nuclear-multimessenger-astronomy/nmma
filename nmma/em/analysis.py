import json
import os
from pathlib import Path
import yaml
from ast import literal_eval
from copy import deepcopy

import bilby
import bilby.core
import matplotlib
import numpy as np
import pandas as pd
from bilby.core.likelihood import ZeroLikelihood

from matplotlib.pyplot import cm

from nmma.em import lightcurve_handling as lch
from .lightcurve_generation import create_light_curve_data
from .em_likelihood import EMTransientLikelihood
from nmma.em import model  
from .prior import create_prior_from_args
from .utils import setup_sample_times, create_detection_limit, get_filtered_mag 
from .io import loadEvent
from .plotting_utils import basic_em_analysis_plot, bolometric_lc_plot
from .em_parsing import parsing_and_logging, em_analysis_parser, bolometric_parser
matplotlib.use("agg")

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


def set_filters(args):
    if args.filters:
        filters = args.filters.replace(" ", "")  # remove all whitespace
        filters = filters.split(",")
        if len(filters) == 0:
            raise ValueError("Need at least one valid filter.")
    elif hasattr(args, "rubin_ToO_type"):
        if args.rubin_ToO_type == 'platinum':
            filters = ["ps1__g","ps1__r","ps1__i","ps1__z","ps1__y"]
        elif args.rubin_ToO_type == 'gold':
            filters = ["ps1__g","ps1__r","ps1__i"]
        elif args.rubin_ToO_type == 'gold_z':
            filters = ["ps1__g","ps1__r","ps1__z"]
        elif args.rubin_ToO_type == 'silver':
            filters = ["ps1__g""ps1__i"]
        elif args.rubin_ToO_type == 'silver_z':
            filters = ["ps1__g","ps1__z"]
    else:
        filters = None
    return filters

def fetch_bestfit(args, light_curve_model, sample_times):
    posterior_file = os.path.join(
        args.outdir, f"{args.label}_posterior_samples.dat"
    )
    posterior_samples = pd.read_csv(posterior_file, header=0, delimiter=" ")
    bestfit_idx = np.argmax(posterior_samples.log_likelihood.to_numpy())
    bestfit_params = posterior_samples.to_dict(orient="list")
    for key in bestfit_params.keys():
        bestfit_params[key] = bestfit_params[key][bestfit_idx]
    print(
        f"Best fit parameters: {str(bestfit_params)}\nBest fit index: {bestfit_idx}"
    )

    obs_sample_times, obs_lightcurve = light_curve_model.gen_detector_lc(
        bestfit_params, sample_times
    )
    if not isinstance(obs_lightcurve, dict): # bolometric model, have to turn it into a dict
        obs_lightcurve = {'lbol': obs_lightcurve}
    obs_lightcurve["bestfit_sample_times"] = obs_sample_times

    bestfit_params["Best fit index"] = int(bestfit_idx)

    return obs_lightcurve, bestfit_params

def make_injection(args, filters = None, fixed_timestep=False, injection_model = None):

    if isinstance(args.injection, str):
        args.injection_file = args.injection
    with open(args.injection_file, "r") as f:
        injection_dict = json.load(
            f, object_hook=bilby.core.utils.decode_bilby_json
        )
    injection_df = injection_dict["injections"]
    injection_parameters = injection_df.iloc[args.injection_num].to_dict()

    injection_parameters = lch.read_trigger_time(injection_parameters, args)
    
    if args.ignore_timeshift:
        injection_parameters.pop('timeshift', None)

    injection_parameters["em_trigger_time"] += injection_parameters.get("timeshift",0)

    if args.prompt_collapse:
        injection_parameters["log10_mej_wind"] = -3.0

    # sanity check for eject masses
    if "log10_mej_dyn" in injection_parameters and not np.isfinite(
        injection_parameters["log10_mej_dyn"]
    ):
        injection_parameters["log10_mej_dyn"] = -3.0
    if "log10_mej_wind" in injection_parameters and not np.isfinite(
        injection_parameters["log10_mej_wind"]
    ):
        injection_parameters["log10_mej_wind"] = -3.0

    if fixed_timestep:
        # FIXME: this is a temporary fix for the time step issue
        # need to interpolate between data points if time step is not 0.25
        if args.em_tstep:
            time_step = args.em_tstep
            if args.em_tstep != 0.25:
                raise ValueError("Need em_tstep to be 0.25 until interpolation feature is incorporated.")
                # currently no linear interpolation function
                do_lin_interpolation = True
            else:
                do_lin_interpolation = False

    if injection_model is None:
        print("Creating injection light curve model")
        injection_model = model.create_injection_model(args, filters)

    sample_times = setup_sample_times(args)
    data = create_light_curve_data(
        injection_parameters, args, 
        light_curve_model=injection_model,
        sample_times=sample_times
    )
    print("Injection generated")

    return data, injection_parameters 

def inspect_detection_limit(detection_limit, data):

    #checking produced data for magnitudes dimmer than the detection limit
    for filt, limit in detection_limit.items():
        for i, row in enumerate(data[filt]):
            mjd, mag, _ = row
            if mag > limit:
                data[filt][i,:] = [mjd, limit, -np.inf]
    return data

def store_injections(detection_limit, filters, data, outfile):
    ref_filts = filters if filters else data.keys()

    data_out = []
    for filt, rows in data.items():
        if filt not in ref_filts:
                continue

        for mjd, mag, mag_unc in rows:
            if not np.isfinite(mag_unc):
                data_out.append([mjd, 99.0, 99.0, filt, mag, 0.0])
            else:
                limmag = detection_limit.get(filt, np.inf)
                data_out.append([mjd, mag, mag_unc, filt, limmag, 0.0])

    data_out = np.array(data_out)

    columns = ["jd", "mag", "mag_unc", "filter", "limmag", "programid"]
    lc = pd.DataFrame(data=data_out, columns=columns)
    lc.sort_values("jd", inplace=True)
    lc = lc.reset_index(drop=True)
    lc.to_csv(outfile)

def check_detections(data, remove_nondetections=False):
    if remove_nondetections:
        data = {filt: data[filt][np.isfinite(data[filt][:, 2])] for filt in data 
                if np.isfinite(data[filt][:, 2]).any()}
    
    if not any(np.isfinite(data[filt][:, 2]).any() 
               and np.isfinite(data[filt][:, 1]).any() 
               for filt in data):
        raise ValueError("Need at least one detection to do fitting.")
    return data

def set_analysis_filters(filters, data):
    if filters is None:
        filters = list(data.keys())

    filters_to_analyze = [filt for filt in data.keys() if filt in filters]
    print(f"Running with filters {filters_to_analyze}")
    return filters_to_analyze

    

def bolometric_analysis(args):

    sample_times = setup_sample_times(args)
    # create the data 
    # FIXME add  injection functionality
    if args.injection:
        pass

    # load the bolometric data
    data = pd.read_csv(args.data)
    light_curve_model = model.SimpleBolometricLightCurveModel(model=args.em_model)

    # setup the prior
    priors = create_prior_from_args(args)

    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        light_curve_data=data,
        priors = priors,
        error_budget=args.error_budget,
        sample_times = sample_times,
        verbose=args.verbose,
    )
    likelihood = EMTransientLikelihood(**likelihood_kwargs)
    result = em_only_sampling(likelihood, priors, args)

    result.plot_corner()

    if args.bestfit or args.plot:
        lbol_dict, _  = fetch_bestfit(args, light_curve_model, sample_times)

        data_times = data["phase"].to_numpy()
        y = data["Lbb"].to_numpy()
        sigma_y = data["Lbb_unc"].to_numpy()
        bolometric_lc_plot(data_times, y, sigma_y, lbol_dict, 
            save_path = os.path.join(args.outdir, f"{args.label}_lightcurves.png")
        )
    return

def analysis(args):
    filters = set_filters(args)
    detection_limit = create_detection_limit(args, filters)

    # create the  data if an injection set is given
    if args.injection:
        data, injection_parameters = make_injection(args, filters)
        data = inspect_detection_limit(detection_limit, data)
        ## configure the output file
        if args.injection_outfile is not None:
            store_injections(detection_limit, filters, data, args.injection_outfile)
        trigger_time = injection_parameters.get('em_trigger_time', None)
        
    else:
        # load observational data
        data = loadEvent(args.data)

        trigger_time = getattr(args, 'em_trigger_time', None)

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
        param_conv = 'AnBa2022'
    # elif to be extended...
    else:
        param_conv = None
    priors = create_prior_from_args(args, param_conv = param_conv)

    sample_times = setup_sample_times(args)
    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters_to_analyze,
        light_curve_data=data,
        priors=priors,
        sample_times = sample_times,
        trigger_time=trigger_time,
        error_budget=args.em_error_budget,
        verbose=args.verbose,
        detection_limit=detection_limit,
        systematics_file=args.systematics_file
    )

    likelihood = EMTransientLikelihood(**likelihood_kwargs)

    result = em_only_sampling(likelihood, priors, args)
    
    if args.injection:
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
        best_mags, bestfit_params = fetch_bestfit(args, light_curve_model, sample_times)
        transient = likelihood.sub_model
        transient.parameters = bestfit_params

        ######################
        # calculate the chi2 #
        ######################
        # fetch data
        processed_times = deepcopy(transient.light_curve_times)
        processed_mags = deepcopy(transient.light_curve_mags)
        processed_uncertainties = deepcopy(transient.light_curve_uncertainties)
        chi2 = 0.0
        dof = 0.0
        chi2_per_dof_dict = {}
        for filt in filters_to_analyze:
            # make best-fit lc interpolation
            mag_used = best_mags[filt]
            t, y, sigma_y = processed_times[filt], processed_mags[filt], processed_uncertainties[filt]
            # print("the time values before adding timeshift are: ", t)
            # # shift t values by timeshift
            # if "timeshift" in bestfit_params:
            #     print("timeshift found in bestfit_params is: ",bestfit_params["timeshift"])
            #     t += bestfit_params["timeshift"]
            # only the detection data are needed
            finite_idx = np.isfinite(sigma_y)
            n_finite = finite_idx.sum()
            print(f"the {filt} data being analyzed is: ", t, y, sigma_y)
            print(f"for {filt} the length of the detections array is: ", n_finite)
            if n_finite > 0:
                t_det, y_det, sigma_y_det = (
                    t[finite_idx],
                    y[finite_idx],
                    sigma_y[finite_idx],
                )
                err = transient.compute_em_err(filt, t_det)
                # print("the time passes into the interp is: ", t_det)
                num = (y_det - np.interp(t_det,best_mags["bestfit_sample_times"], mag_used)) ** 2
                den = sigma_y_det**2 + err**2
                chi2_per_filt = np.sum(num / den)
                # store the data
                chi2 += chi2_per_filt
                dof += n_finite
                print("the number of dof are: ", dof)
                chi2_per_dof_dict[filt] = chi2_per_filt / n_finite

        if dof == 0:
            print("Uh oh! the dof is zero")

        chi2_per_dof = chi2 / dof

    if args.bestfit:
        bestfit_to_write = bestfit_params.copy()
        bestfit_to_write["log_bayes_factor"] = result.log_bayes_factor
        bestfit_to_write["log_bayes_factor_err"] = result.log_evidence_err
        bestfit_to_write["Best fit index"] = bestfit_params["Best fit index"]
        bestfit_to_write["Magnitudes"] = {i: best_mags[i].tolist() for i in best_mags.keys()}
        bestfit_to_write["chi2_per_dof"] = chi2_per_dof
        bestfit_to_write["chi2_per_dof_per_filt"] = {
            i: chi2_per_dof_dict[i].tolist() for i in chi2_per_dof_dict.keys()
        }
        bestfit_file = os.path.join(args.outdir, f"{args.label}_bestfit_params.json")

        with open(bestfit_file, "w") as file:
            json.dump(bestfit_to_write, file, indent=4)

        print(f"Saved bestfit parameters and magnitudes to {bestfit_file}")

    if args.plot:
        filters_to_plot = [
            filt for filt in filters_to_analyze
            if not np.isnan(data[filt][:, 1]).all()
        ]
        mags_to_plot = [get_filtered_mag(best_mags, filt) for filt in filters_to_plot]

        if isinstance(light_curve_model, model.CombinedLightCurveModelContainer):
            sub_models = light_curve_model.models
            model_colors = cm.Spectral(np.linspace(0, 1, len(sub_models)))[::-1]
            obs_times , mag_all = light_curve_model.gen_detector_lc(
                sample_times, bestfit_params, return_all=True
            )
            sub_model_plot_props = {}
            for i, sub_model in enumerate(sub_models):
                sub_model_plot_props[sub_model.model] ={
                    'color': model_colors[i], 
                    'plot_mags' : [get_filtered_mag(mag_all[i], filt) for filt in filters_to_plot],
                    'plot_times': obs_times[i]
                }
        else: sub_model_plot_props = None

        breakpoint()
        
        basic_em_analysis_plot(
            transient, filters_to_plot, mags_to_plot,
            sub_model_plot_props,
            sample_times = best_mags["bestfit_sample_times"],
            xlim = args.xlim, ylim = args.ylim, 
            save_path = os.path.join(args.outdir, f"{args.label}_lightcurves.png")
        )
        breakpoint()
        print('completed')

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
    
    detection_limit = create_detection_limit(args, filters, 22.)

    # create the kilonova data if an injection set is given
    if args.injection:
        data, injection_parameters = make_injection(args, filters, fixed_timestep=0.25)

        if args.injection_outfile is not None:
            store_injections(detection_limit, filters, data, args.injection_outfile)

    else:
        # load the lightcurve data
        # sample_times = setup_sample_times(args)
        data = loadEvent(args.data)
        

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
    t_list = data[res][:, 0]
    data_df = pd.DataFrame({'t': t_list})
    for key in data:
        data_df[key] = data[key][:, 1]
    column_list = data_df.columns.to_list()

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
        
    if args.injection:
        avail_parameters = injection_parameters.keys()
        if ('log10_mej' in avail_parameters) and ('log10_vej' in avail_parameters) and ('log10_Xlan' in avail_parameters):
            param_tensor = torch.tensor([injection_parameters['log10_mej'], injection_parameters['log10_vej'], injection_parameters['log10_Xlan']], dtype=torch.float32)
            with torch.no_grad():
               truth = param_tensor
            flow_result = cast_as_bilby_result(samples, truth, priors=priors)
            fig = flow_result.plot_corner(save=True, label = args.label, outdir=args.outdir)
            print('saved posterior plot')
        else:
            raise ValueError('The injection parameters provided do not match the parameters the flow has been trained on')
    else:
        flow_result = cast_as_bilby_result(samples, truth=None, priors=priors)
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
    args = parsing_and_logging(em_analysis_parser, args)
    if args.sampler == "neuralnet":
        analysis_function = nnanalysis
    else:
        analysis_function = analysis
    multi_analysis_loop(args, analysis_function)
    

def lbol_main(args=None):
    args = parsing_and_logging(bolometric_parser, args)
    multi_analysis_loop(args, bolometric_analysis)
