import os
import matplotlib
import numpy as np
import pandas as pd

from .lightcurve_handling import create_light_curve_data, adjust_injection_parameters
from .em_likelihood import EMTransientLikelihood
from .prior import create_prior_from_args
from . import io, model, utils, systematics  
from .em_parsing import parsing_and_logging, multi_wavelength_analysis_parser, bolometric_parser
from ..core.base import multi_analysis_loop
from ..core.utils import read_injection_file, set_filename, read_trigger_time
matplotlib.use("agg")

def data_from_injection(args, filters, detection_limit):
    injection_df = read_injection_file(args)
    inj_model = model.create_injection_model(args, filters)
    injection_params = injection_df.iloc[args.injection_num].to_dict()
    injection_params = adjust_injection_parameters(injection_params, args,inj_model)
    inj_outfile = set_filename(args.label, args, "_lc")
    if os.path.isfile(inj_outfile):
        print(f"Loading existing injection lc from {inj_outfile}")
        data = io.load_em_observations(inj_outfile, format='model')
    else:
        data = create_light_curve_data(injection_params, args, inj_model)
        io.write_em_observations(inj_outfile, data, format='model')
    data = inspect_detection_limit(detection_limit, data)
    return data, injection_params

def inspect_detection_limit(detection_limit, data):
    #checking produced data for magnitudes dimmer than the detection limit
    for filt, filt_dict in data.items():
        filt_dict = data[filt]
        non_detections = filt_dict['mag'] > detection_limit[filt]

        filt_dict['mag'] = np.where(non_detections, detection_limit[filt], filt_dict['mag'])
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


def bolometric_setup(args):

    # create the data 
    # FIXME add  injection functionality
    # if args.injection_file:
    #     pass
    injection_parameters = None

    # load the bolometric data
    data = pd.read_csv(args.light_curve_data)
    trigger_time = read_trigger_time(None,args)
    light_curve_data = utils.setup_bolometric_lc_data(data, trigger_time)

    light_curve_model = model.SimpleBolometricLightCurveModel(args.em_model,
        sample_times=utils.setup_sample_times(args),  ## usually None, defaults to model_times
    )
    systematics_handler = systematics.SystematicsHandler(
        args.systematics_file, args.em_error_budget, light_curve_data[0])

    # setup the prior
    priors = create_prior_from_args(args, systematics_handler)

    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        light_curve_data=light_curve_data,
        priors=priors,
        systematics_handler=systematics_handler,
        verbose=args.verbose,
    )
    likelihood = EMTransientLikelihood(**likelihood_kwargs)

    return priors, likelihood, injection_parameters

def analysis_setup(args):

    filters = utils.set_filters(args)
    detection_limit = utils.create_detection_limit(args, filters)
        
    try:
        # load observational data
        data = io.load_em_observations(args, format='observations')
        trigger_time = read_trigger_time(None,args)
        injection_parameters = None
    except ValueError:
        # try to work with injection data instead
        data, injection_parameters = data_from_injection(args, filters, detection_limit)
        trigger_time = injection_parameters.get('trigger_time',0)
    except FileNotFoundError:
        # If the injection file is not found, raise an error
        raise FileNotFoundError("Injection file not found.")

    data = utils.cut_data_to_time_range(data, args, trigger_time)
    data = check_detections(data, args.remove_nondetections)
    filters_to_analyze = set_analysis_filters(filters, data)

    # initialize light curve model
    print("Creating light curve model for inference")
    lc_model_type = model.identify_model_type(args)
    light_curve_model = model.create_light_curve_model_from_args(
        lc_model_type, args, filters=filters_to_analyze,
    )
    
    light_curve_data = utils.setup_filtered_lc_data(data, trigger_time)
    systematics_handler = systematics.FilterSystematicsHandler(filters_to_analyze,
        args.systematics_file, args.em_error_budget, light_curve_data[0])
    priors = create_prior_from_args(args, systematics_handler)
    if injection_parameters is not None:
        injection_parameters = {k: injection_parameters.get(k, None) for k in priors.keys()}
    light_curve_data = utils.check_model_time_consistency(light_curve_data, 
                                light_curve_model, priors, injection_parameters)
    # setup the likelihood
    likelihood_kwargs = dict(
        light_curve_model=light_curve_model,
        filters=filters_to_analyze,
        light_curve_data=light_curve_data,
        priors=priors,
        systematics_handler=systematics_handler,
        verbose=args.verbose,
        detection_limit=detection_limit
    )

    likelihood = EMTransientLikelihood(**likelihood_kwargs)
    return priors, likelihood, injection_parameters

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
    systematics_handler = systematics.FilterSystematicsHandler(filters_to_analyze,
        args.systematics_file, error_budget=args.em_error_budget)
    priors = create_prior_from_args(args, systematics_handler)
    
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

def main(args=None):
    args = parsing_and_logging(multi_wavelength_analysis_parser, args)
    if args.sampler == 'neuralnet':
        nnanalysis(args)
    else:
        multi_analysis_loop(args, analysis_setup)
    

def lbol_main(args=None):
    args = parsing_and_logging(bolometric_parser, args)
    multi_analysis_loop(args, bolometric_setup)
