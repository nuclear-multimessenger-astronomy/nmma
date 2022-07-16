import copy
from joblib import load
import numpy as np
import pandas as pd
from importlib import resources
from scipy.interpolate import interp1d

from .model import SVDLightCurveModel, KilonovaGRBLightCurveModel
from .utils import estimate_mag_err


def create_light_curve_data(
    injection_parameters, args, doAbsolute=False, light_curve_model=None
):

    kilonova_kwargs = dict(
        model=args.kilonova_injection_model,
        svd_path=args.kilonova_injection_svd,
        mag_ncoeff=args.injection_svd_mag_ncoeff,
        lbol_ncoeff=args.injection_svd_lbol_ncoeff,
    )

    bands = {1: "g", 2: "r", 3: "i"}
    inv_bands = {v: k for k, v in bands.items()}

    if args.ztf_sampling:
        with resources.open_binary(
            __package__ + ".data", "ZTF_revisit_kde_public.joblib"
        ) as f:
            ztfrevisit = load(f)
        with resources.open_binary(
            __package__ + ".data", "ZTF_sampling_public.pkl"
        ) as f:
            ztfsampling = load(f)
        with resources.open_binary(
            __package__ + ".data", "ZTF_revisit_kde_i.joblib"
        ) as f:
            ztfrevisit_i = load(f)
        with resources.open_binary(__package__ + ".data", "lims_public_g.joblib") as f:
            ztflimg = load(f)
        with resources.open_binary(__package__ + ".data", "lims_public_r.joblib") as f:
            ztflimr = load(f)
        with resources.open_binary(__package__ + ".data", "lims_i.joblib") as f:
            ztflimi = load(f)

    if args.ztf_uncertainties:
        with resources.open_binary(__package__ + ".data", "ZTF_uncer_params.pkl") as f:
            ztfuncer = load(f)
    if args.ztf_ToO:
        with resources.open_binary(
            __package__ + ".data", "sampling_ToO_" + args.ztf_ToO + ".pkl"
        ) as f:
            ztftoo = load(f)
        with resources.open_binary(
            __package__ + ".data", "lims_ToO_" + args.ztf_ToO + "_g.joblib"
        ) as f:
            ztftoolimg = load(f)
        with resources.open_binary(
            __package__ + ".data", "lims_ToO_" + args.ztf_ToO + "_r.joblib"
        ) as f:
            ztftoolimr = load(f)

    tc = injection_parameters["kilonova_trigger_time"]
    tmin = args.kilonova_tmin
    tmax = args.kilonova_tmax
    tstep = args.kilonova_tstep
    if args.injection_detection_limit is None:
        detection_limit = {x: np.inf for x in args.filters.split(",")}
    else:
        detection_limit = {
            x: float(y)
            for x, y in zip(
                args.filters.split(","), args.injection_detection_limit.split(",")
            )
        }
    seed = args.generation_seed

    np.random.seed(seed)

    sample_times = np.arange(tmin, tmax + tstep, tstep)
    Ntimes = len(sample_times)

    if light_curve_model is None:
        if args.with_grb_injection:
            light_curve_model = KilonovaGRBLightCurveModel(
                sample_times=sample_times,
                kilonova_kwargs=kilonova_kwargs,
                GRB_resolution=np.inf,
            )

        else:
            light_curve_model = SVDLightCurveModel(
                sample_times=sample_times, gptype=args.gptype, **kilonova_kwargs
            )

    lbol, mag = light_curve_model.generate_lightcurve(
        sample_times, injection_parameters
    )
    dmag = args.kilonova_error

    data = {}

    if args.ztf_sampling or args.rubin_ToO or args.photometry_augmentation:
        passbands_to_keep = []

    for filt in mag:
        mag_per_filt = mag[filt]
        if filt in detection_limit:
            det_lim = detection_limit[filt]
        elif (
            args.photometry_augmentation_filters is not None
            and filt in args.photometry_augmentation_filters.split(",")
        ):
            det_lim = 30.0
        else:
            det_lim = np.inf

        if not doAbsolute:
            if injection_parameters["luminosity_distance"] > 0:
                mag_per_filt += 5.0 * np.log10(
                    injection_parameters["luminosity_distance"] * 1e6 / 10.0
                )
        data_per_filt = np.zeros([Ntimes, 3])
        for tidx in range(0, Ntimes):
            if mag_per_filt[tidx] >= det_lim:
                data_per_filt[tidx] = [sample_times[tidx] + tc, det_lim, np.inf]
            else:
                noise = np.random.normal(scale=dmag)
                if args.ztf_uncertainties and filt in ["g", "r", "i"]:
                    df = pd.DataFrame.from_dict(
                        {
                            "passband": [inv_bands[filt]],
                            "mag": [mag_per_filt[tidx] + noise],
                        }
                    )
                    df = estimate_mag_err(ztfuncer, df)
                    if not df["mag_err"].values:
                        data_per_filt[tidx] = [sample_times[tidx] + tc, det_lim, np.inf]
                    else:
                        data_per_filt[tidx] = [
                            sample_times[tidx] + tc,
                            mag_per_filt[tidx] + noise,
                            df["mag_err"].values[0],
                        ]
                else:
                    data_per_filt[tidx] = [
                        sample_times[tidx] + tc,
                        mag_per_filt[tidx] + noise,
                        dmag,
                    ]
        data[filt] = data_per_filt

    data_original = copy.deepcopy(data)
    if args.ztf_sampling:
        sim = pd.DataFrame()
        start = np.random.uniform(tc, tc + 2)
        t = start
        # ZTF-II Public
        while t < tmax + tc:
            sample = ztfsampling.sample()
            sim = pd.concat(
                [
                    sim,
                    pd.DataFrame(
                        np.array(
                            [t + sample["t"].values[0], sample["bands"].values[0]]
                        ).T
                    ),
                ]
            )
            t += float(ztfrevisit.sample())
        # i-band
        start = np.random.uniform(tc, tc + 4)
        t = start
        while t < tmax + tc:
            sim = pd.concat([sim, pd.DataFrame([[t, 3]])])
            t += float(ztfrevisit_i.sample())
        sim["ToO"] = False
        # toO
        if args.ztf_ToO:
            sim_ToO = pd.DataFrame()
            start = np.random.uniform(tc, tc + 1)
            t = start
            too_samps = ztftoo.sample(np.random.choice([1, 2]))
            for i, too in too_samps.iterrows():
                sim_ToO = pd.concat(
                    [sim_ToO, pd.DataFrame(np.array([t + too["t"], too["bands"]]).T)]
                )
                t += 1
            sim_ToO["ToO"] = True
            sim = pd.concat([sim, sim_ToO])

        sim = (
            sim.rename(columns={0: "mjd", 1: "passband"})
            .sort_values(by=["mjd"])
            .reset_index(drop=True)
        )
        sim["passband"] = sim["passband"].map({1: "g", 2: "r", 3: "i"})
        sim["mag"] = np.nan
        sim["mag_err"] = np.nan

        for filt, group in sim.groupby("passband"):
            data_per_filt = copy.deepcopy(data_original[filt])
            lc = interp1d(
                data_per_filt[:, 0],
                data_per_filt[:, 1],
                fill_value=np.inf,
                bounds_error=False,
                assume_sorted=True,
            )
            lc_err = interp1d(
                data_per_filt[:, 0],
                data_per_filt[:, 2],
                fill_value=np.inf,
                bounds_error=False,
                assume_sorted=True,
            )
            sim.loc[group.index, "mag"] = lc(group["mjd"].tolist())
            sim.loc[group.index, "mag_err"] = lc_err(group["mjd"].tolist())

        for filt, group in sim.groupby("passband"):
            if args.ztf_uncertainties and filt in ["g", "r", "i"]:
                mag_err = []
                for idx, row in group.iterrows():
                    upperlimit = False
                    if filt == "g" and row["ToO"] is False:
                        limg = float(ztflimg.sample())
                        if row["mag"] > limg:
                            sim.loc[row.name, "mag"] = limg
                            sim.loc[row.name, "mag_err"] = np.inf
                    elif filt == "g" and row["ToO"] is True:
                        toolimg = float(ztftoolimg.sample())
                        if row["mag"] > toolimg:
                            sim.loc[row.name, "mag"] = toolimg
                            sim.loc[row.name, "mag_err"] = np.inf
                    elif filt == "r" and row["ToO"] is False:
                        limr = float(ztflimr.sample())
                        if row["mag"] > limr:
                            sim.loc[row.name, "mag"] = limr
                            sim.loc[row.name, "mag_err"] = np.inf
                    elif filt == "r" and row["ToO"] is True:
                        toolimr = float(ztftoolimr.sample())
                        if row["mag"] > toolimr:
                            sim.loc[row.name, "mag"] = toolimr
                            sim.loc[row.name, "mag_err"] = np.inf
                    else:
                        limi = float(ztflimi.sample())
                        if row["mag"] > limi:
                            sim.loc[row.name, "mag"] = limi
                            sim.loc[row.name, "mag_err"] = np.inf
                    if not np.isfinite(sim.loc[row.name, "mag_err"]):
                        upperlimit = True
                    if upperlimit:
                        mag_err.append(np.inf)
                    else:
                        df = pd.DataFrame.from_dict(
                            {"passband": [filt], "mag": [sim.loc[row.name, "mag"]]}
                        )
                        df["passband"] = df["passband"].map(
                            {"g": 1, "r": 2, "i": 3}
                        )  # estimate_mag_err maps filter numbers
                        df = estimate_mag_err(ztfuncer, df)
                        sim.loc[row.name, "mag_err"] = float(df["mag_err"])
                        mag_err.append(df["mag_err"].tolist()[0])

                data_per_filt = np.vstack(
                    [
                        sim.loc[group.index, "mjd"].tolist(),
                        sim.loc[group.index, "mag"].tolist(),
                        mag_err,
                    ]
                ).T
            else:
                data_per_filt = np.vstack(
                    [
                        sim.loc[group.index, "mjd"].tolist(),
                        sim.loc[group.index, "mag"].tolist(),
                        sim.loc[group.index, "mag_err"].tolist(),
                    ]
                ).T
            data[filt] = data_per_filt
            passbands_to_keep.append(filt)
        if args.train_stats:
            sim["tc"] = tc
            sim.to_csv(args.outdir + "/too.csv", index=False)

    if args.rubin_ToO:
        start = tmin + tc
        if args.rubin_ToO_type == "BNS":
            strategy = [
                [1 / 24.0, ["u", "g", "r", "i", "z", "y"]],
                [2 / 24.0, ["u", "g", "r", "i", "z", "y"]],
                [4 / 24.0, ["u", "g", "r", "i", "z", "y"]],
                [1.0, ["g", "z", "i"]],
            ]
        elif args.rubin_ToO_type == "NSBH":
            strategy = [
                [1 / 24.0, ["u", "g", "r", "i", "z", "y"]],
                [4 / 24.0, ["u", "g", "r", "i", "z", "y"]],
                [1.0, ["u", "g", "r", "i", "z", "y"]],
                [2.0, ["g", "z", "i"]],
            ]
        else:
            raise ValueError("args.rubin_ToO_type should be either BNS or NSBH")

        mjds, passbands = [], []
        sim = pd.DataFrame()
        for (obstime, filts) in strategy:
            for filt in filts:
                mjds.append(tc + obstime)
                passbands.append(filt)
        sim = pd.DataFrame.from_dict({"mjd": mjds, "passband": passbands})

        for filt, group in sim.groupby("passband"):
            data_per_filt = copy.deepcopy(data_original[filt])
            lc = interp1d(
                data_per_filt[:, 0],
                data_per_filt[:, 1],
                fill_value=np.inf,
                bounds_error=False,
                assume_sorted=True,
            )
            lcerr = interp1d(
                data_per_filt[:, 0],
                data_per_filt[:, 2],
                fill_value=np.inf,
                bounds_error=False,
                assume_sorted=True,
            )
            times = group["mjd"].tolist()
            data_per_filt = np.vstack([times, lc(times), lcerr(times)]).T
            data[filt] = data_per_filt
            passbands_to_keep.append(filt)

    if args.photometry_augmentation:
        np.random.seed(args.photometry_augmentation_seed)
        if args.photometry_augmentation_filters is None:
            filts = np.random.choice(
                list(data.keys()),
                size=args.photometry_augmentation_N_points,
                replace=True,
            )
        else:
            filts = args.photometry_augmentation_filters.split(",")

        if args.photometry_augmentation_times is None:
            tt = np.random.uniform(
                tmin + tc, tmax + tc, size=args.photometry_augmentation_N_points
            )
        else:
            tt = tc + np.array(
                [float(x) for x in args.photometry_augmentation_times.split(",")]
            )

        for filt in list(set(filts)):
            data_per_filt = copy.deepcopy(data_original[filt])
            idx = np.where(filt == filts)[0]
            times = tt[idx]

            if len(times) == 0:
                continue

            lc = interp1d(
                data_per_filt[:, 0],
                data_per_filt[:, 1],
                fill_value=np.inf,
                bounds_error=False,
                assume_sorted=True,
            )
            lcerr = interp1d(
                data_per_filt[:, 0],
                data_per_filt[:, 2],
                fill_value=np.inf,
                bounds_error=False,
                assume_sorted=True,
            )
            data_per_filt = np.vstack([times, lc(times), lcerr(times)]).T
            if filt not in data:
                data[filt] = data_per_filt
            else:
                data[filt] = np.vstack([data[filt], data_per_filt])
                data[filt] = data[filt][data[filt][:, 0].argsort()]
            passbands_to_keep.append(filt)

    if args.ztf_sampling or args.rubin_ToO or args.photometry_augmentation:
        passbands_to_lose = set(list(data.keys())) - set(passbands_to_keep)
        for filt in passbands_to_lose:
            del data[filt]

    return data
