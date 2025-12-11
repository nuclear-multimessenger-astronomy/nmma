import bilby
import bilby.core
from bilby.core.prior import Prior
from bilby.core.prior import PriorDict as _PriorDict
from bilby.core.prior.interpolated import Interped
from bilby.core.prior.conditional import ConditionalTruncatedGaussian

from . import systematics


def from_list(self, systematics):
    """
    Similar to `from_file` but instead of file buffer, takes a list of Prior strings
    See `from_file` for more details
    """

    comments = ["#", "\n"]
    prior = dict()
    for line in systematics:
        if line[0] in comments:
            continue
        line.replace(" ", "")
        elements = line.split("=")
        key = elements[0].replace(" ", "")
        val = "=".join(elements[1:]).strip()
        prior[key] = val
    self.from_dictionary(prior)


setattr(_PriorDict, "from_list", from_list)


class ConditionalGaussianIotaGivenThetaCore(ConditionalTruncatedGaussian):
    """
    This prior characterizes the conditional prior on the viewing angle
    of a GRB afterglow given the opening angle of the jet such that
    the viewing angle is follwing a half-Gaussian distribution
    with the opening angle (thetaCore) as the sigma;
    therefore, sigma = thetaCore / N_sigma
    a higher value of N_sigma, the distribution would be tigher around 0, vice versa

    This prior should only be used if the event is a prompt emission!
    """

    def __init__(
        self,
        minimum,
        maximum,
        name,
        N_sigma=1,
        latex_label=None,
        unit=None,
        boundary=None,
    ):

        super(ConditionalTruncatedGaussian, self).__init__(
            mu=0,
            sigma=1,
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self._condition_function,
        )

        self._required_variables = ["thetaCore"]
        self.N_sigma = N_sigma
        self.__class__.__name__ = "ConditionalGaussianIotaGivenThetaCore"
        self.__class__.__qualname__ = "ConditionalGaussianIotaGivenThetaCore"

    def _condition_function(self, reference_params, **kwargs):
        return dict(sigma=(1.0 / self.N_sigma) * kwargs[self._required_variables[0]])

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


def inclination_prior_from_fits(priors, args):
    from ligo.skymap import io, moc
    from scipy.interpolate import PchipInterpolator
    from scipy.stats import norm
    import healpy as hp
    import numpy as np
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    print("Constructing prior on inclination with fits input")

    # load the skymap
    skymap = io.read_sky_map(args.fits_file, moc=True)

    # check if the sky location is input
    # if not, the maximum posterior point is taken
    if ("ra" in priors and "dec" in priors) or (args.ra and args.dec):
        if args.ra and args.dec:
            print(
                "Using command line input for sky location, ignoring the prior file input"
            )
            ra = args.ra
            dec = args.dec
        else:
            print("Using prior file input for sky location")
            ra = np.rad2deg(priors["ra"].peak)
            dec = np.rad2deg(priors["dec"].peak)
        print(f"Using the input sky location ra={ra}, dec={dec}")
        # convert them back to theta and phi
        phi = np.deg2rad(ra)
        theta = 0.5 * np.pi - np.deg2rad(dec)
        # make use of the maP nside
        maP_idx = np.argmax(skymap["PROBDENSITY"])
        order, _ = moc.uniq2nest(skymap[maP_idx]["UNIQ"])
        nside = hp.order2nside(order)
        # get the nested idx for the given sky location
        nest_idx = hp.ang2pix(nside, theta, phi, nest=True)
        # find the row with the closest nested index
        nest_idxs = []
        for row in skymap:
            order_per_row, nest_idx_per_row = moc.uniq2nest(row["UNIQ"])
            if order_per_row == order:
                nest_idxs.append(nest_idx_per_row)
            else:
                nest_idxs.append(0)
        nest_idxs = np.array(nest_idxs)
        row = skymap[np.argmin(np.absolute(nest_idxs - nest_idx))]
    else:
        print("Using the maP point from the fits file input")
        maP_idx = np.argmax(skymap["PROBDENSITY"])
        uniq_idx = skymap[maP_idx]["UNIQ"]
        # convert to nested indexing and find the location of that index
        order, nest_idx = moc.uniq2nest(uniq_idx)
        nside = hp.order2nside(order)
        theta, phi = hp.pix2ang(nside, int(nest_idx), nest=True)
        # convert theta and phi to ra and dec
        ra = np.rad2deg(phi)
        dec = np.rad2deg(0.5 * np.pi - theta)
        print(f"The maP location is ra={ra}, dec={dec}")
        # fetching the skymap row
        row = skymap[maP_idx]

    # construct the iota prior
    cosiota_nodes_num = args.cosiota_node_num
    cosiota_nodes = np.cos(np.linspace(0, np.pi, cosiota_nodes_num))
    colnames = ["PROBDENSITY", "DISTMU", "DISTSIGMA", "DISTNORM"]
    # do an all-in-one interpolation
    prob_density, dist_mu, dist_sigma, dist_norm = (
        PchipInterpolator(
            cosiota_nodes[::-1],
            row["{}_SAMPLES".format(colname)][::-1],
        )
        for colname in colnames
    )
    # now have the joint distribution evaluated
    u = np.linspace(-1, 1, 1000)  # this is cosiota
    # fetch the fixed distance
    if args.dL:
        print("Using command line input for distance, ignoring the prior file input")
        dL = args.dL
    else:
        print("Using prior file input for distance")
        dL = priors["luminosity_distance"].peak
    prob_u = (
        prob_density(u)
        * dist_norm(u)
        * np.square(dL)
        * norm(dist_mu(u), dist_sigma(u)).pdf(dL)
    )

    iota = np.arccos(u)
    prob_iota = prob_u * np.absolute(np.sin(iota))
    # in GW, iota in [0, pi], but in EM, iota in [0, pi/2]
    # therefore, we need to do a folding
    # split the domain in half
    iota_lt_pi2 = iota[iota < np.pi / 2]
    prob_lt_pi2, prob_gt_pi2 = prob_iota[iota < np.pi / 2], prob_iota[iota >= np.pi / 2]
    iota_EM = iota_lt_pi2
    prob_iota_EM = prob_lt_pi2 + prob_gt_pi2[::-1]

    # normalize
    prob_iota /= np.trapz(iota_EM, prob_iota_EM)

    priors["inclination_EM"] = Interped(
        xx=iota_EM, yy=prob_iota_EM, minimum=0, maximum=np.pi / 2
    )

    if args.plot:
        figIdx = np.random.randint(1000)
        plt.figure(figIdx)
        plt.xlabel("Inclination")
        plt.ylabel("PDF")
        plt.plot(iota_EM, prob_iota_EM)
        plt.savefig(f"{args.outdir}/Fits_motivated_inclination_prior.png")

    return priors


def create_prior_from_args(model_names, args):

    AnBa2022_intersect = list(
        set(["AnBa2022_linear", "AnBa2022_log"]) & set(model_names)
    )

    if len(AnBa2022_intersect) > 0:

        def convert_mtot_mni(parameters):

            for param in ["mni", "mtot", "mrp"]:
                if param not in parameters:
                    parameters[param] = 10 ** parameters[f"log10_{param}"]

            parameters["mni_c"] = parameters["mni"] / parameters["mtot"]
            parameters["mrp_c"] = (
                parameters["xmix"] * (parameters["mtot"] - parameters["mni"])
                - parameters["mrp"]
            )
            return parameters

        priors = bilby.core.prior.PriorDict(
            args.prior, conversion_function=convert_mtot_mni
        )
    else:
        priors = bilby.gw.prior.PriorDict(args.prior)

    # setup for Ebv
    if "Ebv" not in priors:
        if args.Ebv_max > 0.0 and args.use_Ebv:
            Ebv_c = 1.0 / (0.5 * args.Ebv_max)
            priors["Ebv"] = bilby.core.prior.Interped(
                name="Ebv",
                minimum=0.0,
                maximum=args.Ebv_max,
                latex_label="$E(B-V)$",
                xx=[0, args.Ebv_max],
                yy=[Ebv_c, 0],
            )
        else:
            priors["Ebv"] = bilby.core.prior.DeltaFunction(
                name="Ebv", peak=0.0, latex_label="$E(B-V)$"
            )

    # re-setup the prior if the conditional prior for inclination is used

    if args.conditional_gaussian_prior_thetaObs:
        priors_dict = dict(priors)
        original_iota_prior = priors_dict["inclination_EM"]
        setup = dict(
            minimum=original_iota_prior.minimum,
            maximum=original_iota_prior.maximum,
            name=original_iota_prior.name,
            latex_label=original_iota_prior.latex_label,
            unit=original_iota_prior.unit,
            boundary=original_iota_prior.boundary,
            N_sigma=args.conditional_gaussian_prior_N_sigma,
        )

        priors_dict["inclination_EM"] = ConditionalGaussianIotaGivenThetaCore(**setup)
        priors = bilby.gw.prior.ConditionalPriorDict(priors_dict)

    if args.fits_file:
        priors = inclination_prior_from_fits(priors, args)

    if args.fetch_Ebv_from_dustmap:
        assert (
            args.ra and args.dec
        ), "sky location is needed for fetching Ebv from dustmap"
        print(
            "Fetching value of Ebv from dustmap, overwriting the original prior on Ebv"
        )
        try:
            import dustmaps.sfd
            from dustmaps.config import config
            from astropy import coordinates
        except ImportError:
            print("Package dustmap is needed")
            import sys

            sys.exit(1)

        # check if the dust fits are downloaded
        import os

        default_dir = os.path.join(os.path.dirname(__file__))
        # config dustmaps to fetch and place fits from default_dir
        config["data_dir"] = default_dir
        required_files = ["sfd/SFD_dust_4096_ngp.fits", "sfd/SFD_dust_4096_sgp.fits"]
        if any(
            [not os.path.isfile(os.path.join(default_dir, f)) for f in required_files]
        ):
            dustmaps.sfd.fetch()
        # fetching the Ebv value
        coord = coordinates.SkyCoord(args.ra, args.dec, unit="deg")
        priors["Ebv"] = float(dustmaps.sfd.SFDQuery()(coord))

        print(f"The prior on Ebv is set to fixed value of {priors['Ebv']}")

    if args.systematics_file is not None:
        systematics_priors = systematics.main(args.systematics_file)
        priors.from_list(systematics_priors)
    return priors
