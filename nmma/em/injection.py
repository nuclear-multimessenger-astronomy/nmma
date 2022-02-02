import copy
from joblib import dump, load
import numpy as np
import pandas as pd
from importlib_resources import files
from scipy.interpolate import interp1d

from .model import SVDLightCurveModel, KilonovaGRBLightCurveModel
from .utils import estimate_mag_err

def create_light_curve_data(injection_parameters, args, doAbsolute=False,
                            light_curve_model=None):

    kilonova_kwargs = dict(model=args.kilonova_injection_model,
                           svd_path=args.kilonova_injection_svd,
                           mag_ncoeff=args.injection_svd_mag_ncoeff,
                           lbol_ncoeff=args.injection_svd_lbol_ncoeff)

    bands = {1: 'g', 2: 'r', 3: 'i'}
    inv_bands = {v: k for k, v in bands.items()}

    if args.ztf_sampling:
        ztfrevisitfile = files('nmma.em.data').joinpath('ZTF_revisit_kde_public.joblib')
        ztfsamplingfile = files('nmma.em.data').joinpath('ZTF_sampling_public.pkl')
        ztfrevisit = load(ztfrevisitfile)
        ztfsampling = load(ztfsamplingfile)

        ztfrevisitifile = files('nmma.em.data').joinpath('ZTF_revisit_kde_i.joblib')
        ztfrevisit_i = load(ztfrevisitfile)

    if args.ztf_uncertainties:
        ztfuncer = load(files('nmma.em.data').joinpath('ZTF_uncer_params.pkl'))


    tc = injection_parameters['kilonova_trigger_time']
    tmin = args.kilonova_tmin
    tmax = args.kilonova_tmax
    tstep = args.kilonova_tstep
    if args.injection_detection_limit is None:
        detection_limit = {x: np.inf for x in args.filters.split(",")}
    else:
        detection_limit = {x: float(y) for x, y in zip(args.filters.split(","), args.injection_detection_limit.split(","))}
    filters = args.filters
    seed = args.generation_seed

    np.random.seed(seed)

    sample_times = np.arange(tmin, tmax + tstep, tstep)
    Ntimes = len(sample_times)

    if light_curve_model is None:
        if args.with_grb_injection:
            light_curve_model = KilonovaGRBLightCurveModel(sample_times=sample_times,
                                                           kilonova_kwargs=kilonova_kwargs,
                                                           GRB_resolution=np.inf)

        else:
            light_curve_model = SVDLightCurveModel(sample_times=sample_times,
                                                   gptype=args.gptype,
                                                   **kilonova_kwargs)

    lbol, mag = light_curve_model.generate_lightcurve(sample_times, injection_parameters)
    dmag = args.kilonova_error

    data = {}

    for filt in mag:
        mag_per_filt = mag[filt]
        if filt in detection_limit:
            det_lim = detection_limit[filt]
        elif args.optimal_augmentation_filters is not None and filt in args.optimal_augmentation_filters.split(","):
            det_lim = 30.0
        else:
            det_lim = 0.0

        if not doAbsolute:
            mag_per_filt += 5. * np.log10(injection_parameters['luminosity_distance'] * 1e6 / 10.)
        data_per_filt = np.zeros([Ntimes, 3])
        for tidx in range(0, Ntimes):
            if mag_per_filt[tidx] >= det_lim:
                data_per_filt[tidx] = [sample_times[tidx] + tc, det_lim, np.inf]
            else:
                noise = np.random.normal(scale=dmag)
                if args.ztf_uncertainties and filt in ['g', 'r', 'i']:
                    df = pd.DataFrame.from_dict({'passband': [inv_bands[filt]], 'mag': [mag_per_filt[tidx] + noise]})
                    df = estimate_mag_err(ztfuncer, df)
                    if not df['mag_err'].values:
                        data_per_filt[tidx] = [sample_times[tidx] + tc, det_lim, np.inf]
                    else:
                        data_per_filt[tidx] = [sample_times[tidx] + tc, mag_per_filt[tidx] + noise, df['mag_err'].values[0]]
                else:
                    data_per_filt[tidx] = [sample_times[tidx] + tc, mag_per_filt[tidx] + noise, dmag]
        data[filt] = data_per_filt

    data_original = copy.deepcopy(data)
    if args.ztf_sampling:
        sim=pd.DataFrame()
        start=np.random.uniform(tc,tc+2)
        t=start
        #ZTF-II Public
        while t<tmax+tc:
            sample=ztfsampling.sample()
            sim=pd.concat([sim,pd.DataFrame(
                                      np.array([t+sample['t'].values[0],
                                      sample['bands'].values[0]]).T
                                      )
                          ])
            t+=float(ztfrevisit.sample())
        #i-band
        start=np.random.uniform(tc,tc+4)
        t=start
        while t<tmax+tc:
            sim=pd.concat([sim,pd.DataFrame([[t,3]])])
            t+=float(ztfrevisit_i.sample())
        #toO
        if args.ztf_ToO:
          ztftoofile = load(files('nmma.em.data').joinpath('sampling_toO_'+args.ztf_ToO+'.pkl'))
          start=np.random.uniform(tc,tc+1)
          t=start
          too_samps=ztftoofile.sample(np.random.choice([1,2]))
          for i,too in too_samps.iterrows():
            sim=pd.concat([sim,pd.DataFrame(
                                      np.array([t+too['t'],too['bands']]).T
                                      )
                          ])
            t+=1

        sim=sim.rename(columns={0:'mjd',1:'passband'}).sort_values(by=['mjd'])
        sim['passband']=sim['passband'].map({1:'g',2:'r',3:'i'})

        for filt,group in sim.groupby('passband'):
            data_per_filt = copy.deepcopy(data_original[filt])
            lc = interp1d(data_per_filt[:,0], data_per_filt[:,1],
                          fill_value=np.inf, bounds_error=False, assume_sorted=True)
            lcerr = interp1d(data_per_filt[:,0], data_per_filt[:,2],
                          fill_value=np.inf, bounds_error=False, assume_sorted=True)
            times=group['mjd'].tolist()
            data_per_filt = np.vstack([times, lc(times), lcerr(times)]).T
            data[filt] = data_per_filt

    if args.optimal_augmentation:
        np.random.seed(args.optimal_augmentation_seed)
        if args.optimal_augmentation_filters is None:
            filts = np.random.choice(list(data.keys()),
                                     size=args.optimal_augmentation_N_points,
                                     replace=True)
        else:
            filts = np.random.choice(args.optimal_augmentation_filters.split(","),
                                     size=args.optimal_augmentation_N_points,
                                     replace=True)
        tt = np.random.uniform(tmin + tc, tmax + tc,
                               size=args.optimal_augmentation_N_points)
        for filt in list(set(filts)):
            data_per_filt = copy.deepcopy(data_original[filt])
            idx = np.where(filt == filts)[0]
            times = tt[idx]

            if len(times) == 0: continue

            lc = interp1d(data_per_filt[:,0], data_per_filt[:,1],
                          fill_value=np.inf, bounds_error=False, assume_sorted=True)
            lcerr = interp1d(data_per_filt[:,0], data_per_filt[:,2],
                          fill_value=np.inf, bounds_error=False, assume_sorted=True)
            data_per_filt = np.vstack([times, lc(times), lcerr(times)]).T
            if filt not in data:
                data[filt] = data_per_filt
            else:
                data[filt] = np.vstack([data[filt], data_per_filt])
                data[filt] = data[filt][data[filt][:,0].argsort()]

    return data
