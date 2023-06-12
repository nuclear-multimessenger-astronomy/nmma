import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize_scalar
from .tov import TOVSolver
import lal


class EOS_with_CSE(object):
    """
    Create and eos object with an array of (n, p, e) as the
    low-density tail. And extend the eos to higher density with
    speed-of-sound interpolation. And with the corresponding
    (m, r, lambda) relation solved.

    Parameters:
        low_density_eos: dict, with numpy arrays of n, p, and e in fm^-3, MeV fm^-3 and MeV fm^-3
        n_connect: float, take the low density eos up to the given number density (default: 0.16)
        n_lim: float, having the eos extend to the given number density (default: 2)
        N_seg: int, number of speed-of-sound extension segments (default: 5)
        cs2_limt: float, speed-of-sound squared limit in c^2 (default: 1)
        seed: int, seed of random draw extension (default: 42)
    """

    def __init__(self, low_density_eos, n_connect=0.16, n_lim=2., N_seg=5, cs2_limit=1., seed=42, extension_scheme='peter', low_density_eos_stiff=None):

        self.seed = seed

        if not low_density_eos_stiff:
            self.n_low = low_density_eos['n']
            self.p_low = low_density_eos['p']
            self.e_low = low_density_eos['e']

        else:
            assert len(low_density_eos) == len(low_density_eos_stiff), ('This requires '
                                                                        'interpolation. '
                                                                        'Will be added '
                                                                        'in the future.')
            # fix the seed
            np.random.seed(self.seed)
            alpha = np.random.uniform()

            self.n_low_soft = low_density_eos['n']
            self.p_low_soft = low_density_eos['p']
            self.e_low_soft = low_density_eos['e']

            self.n_low_stiff = low_density_eos_stiff['n']
            self.p_low_stiff = low_density_eos_stiff['p']
            self.e_low_stiff = low_density_eos_stiff['e']

            diff_e = self.e_low_stiff - self.e_low_soft
            diff_p = self.p_low_stiff - self.p_low_soft

            self.n_low = self.n_low_soft
            self.e_low = self.e_low_soft + alpha * diff_e
            self.p_low = self.p_low_soft + alpha * diff_p

        log_e_of_log_n_low = CubicSpline(np.log(self.n_low), np.log(self.e_low))
        log_p_of_log_n_low = CubicSpline(np.log(self.n_low), np.log(self.p_low))
        log_p_of_log_e_low = CubicSpline(np.log(self.e_low), np.log(self.p_low))

        self.e_at_n_connect = np.exp(log_e_of_log_n_low(np.log(n_connect)))
        self.p_at_n_connect = np.exp(log_p_of_log_n_low(np.log(n_connect)))
        self.cs2_at_n_connect = self.p_at_n_connect / self.e_at_n_connect *\
            log_p_of_log_e_low.derivative()(np.log(self.e_at_n_connect))

        self.n_connect = n_connect
        self.n_lim = n_lim
        self.n_extend_range = n_lim - n_connect
        self.N_seg = N_seg
        self.cs2_limit = cs2_limit

        if extension_scheme == 'peter':
            self.__extend()
        elif extension_scheme == 'rahul':
            self.__extend_v1()

        self.__calculate_pseudo_enthalpy()
        self.__construct_all_interpolation()

    def __extend(self):

        # declare the cs2 on nodes
        cs2_draw = np.empty((self.N_seg + 2, 2))
        # first node is the end of the low density eos
        cs2_draw[0, :] = [self.n_connect, self.cs2_at_n_connect]

        # fix the seed
        np.random.seed(self.seed)

        # draw cs2(n_node) randomly
        for node_index in range(1, self.N_seg + 1):
            n_val_lower_bound = cs2_draw[node_index - 1, 0]
            n_val_upper_bound = min(n_val_lower_bound + 1.5 * self.n_extend_range / self.N_seg, self.n_lim)
            n_val = np.random.uniform(n_val_lower_bound, n_val_upper_bound)
            cs2_val = np.random.uniform(0., self.cs2_limit)
            cs2_draw[node_index] = [n_val, cs2_val]

        # add the speed for sound value at n_lim
        cs2_at_n_lim = np.random.uniform(0., self.cs2_limit)
        cs2_draw[-1] = [self.n_lim, cs2_at_n_lim]

        # interpolation for cs2(n)
        cs2_extent = interp1d(cs2_draw[:, 0], cs2_draw[:, 1], kind='linear',
                              fill_value="extrapolate")

        # construct the extended EOS
        # do the integration in log-space for stability
        n_step = 1e-3
        n_high = np.arange(self.n_connect, self.n_lim, n_step)

        def dydt(t, y, cs2):
            logp, loge = y
            logn = t

            p = np.exp(logp)
            e = np.exp(loge)
            n = np.exp(logn)

            # dedn = (p + e) / n
            # dpdn = cs2(n) * dedn

            dloge_dlogn = 1. + p / e
            dlogp_dlogn = cs2(n) * (e / p + 1.)

            return [dlogp_dlogn, dloge_dlogn]

        y0 = (np.log(self.p_at_n_connect), np.log(self.e_at_n_connect))

        sol = solve_ivp(dydt, (np.log(self.n_connect), np.log(self.n_lim)), y0,
                        method='DOP853', t_eval=np.log(n_high), args=(cs2_extent,), rtol=1e-7, atol=0.)
        logp_high = sol.y[0]
        loge_high = sol.y[1]

        self.n_high = n_high
        self.p_high = np.exp(logp_high)
        self.e_high = np.exp(loge_high)

        n_low = self.n_low
        self.n_array = np.concatenate((self.n_low[n_low < self.n_connect], self.n_high))
        self.p_array = np.concatenate((self.p_low[n_low < self.n_connect], self.p_high))
        self.e_array = np.concatenate((self.e_low[n_low < self.n_connect], self.e_high))

    def __extend_v1(self):

        # fix the seed
        np.random.seed(self.seed)

        n_ext_grid = np.linspace(self.n_connect + 1e-4 * self.n_connect, self.n_lim,
                                 num=self.N_seg + 1)
        c2_ext_grid = [np.random.uniform(0, self.cs2_limit) for i in n_ext_grid]
        c2_ext_grid[0] = self.cs2_at_n_connect

        # Empty grid for the chemical potential corresponding to n_ext_grid
        mu_ext_grid = np.zeros_like(n_ext_grid)
        mu_ext_grid[0] = self.mu_at_n_connect

        num = 50
        n_high = [np.linspace(n_ext_grid[i], n_ext_grid[i + 1], endpoint=False, num=num)
                  for i in range(n_ext_grid.size - 1)]
        n_high = np.array(n_high)
        c2_high = np.zeros_like(n_high)
        mu_high = np.zeros_like(n_high)

        # Integrates the sound speed to compute the chemical potential
        # Fills in all elements of mu_ext
        for i in range(n_ext_grid.size - 1):
            slope = (c2_ext_grid[i + 1] - c2_ext_grid[i]) / (n_ext_grid[i + 1] - n_ext_grid[i])
            c2_high[i, :] = slope * (n_high[i, :] - n_ext_grid[i]) + c2_ext_grid[i]
            mu_ext_grid[i + 1] = mu_ext_grid[i] * np.exp(slope * (n_ext_grid[i + 1] - n_ext_grid[i] - n_ext_grid[i] * np.log(n_ext_grid[i + 1] / n_ext_grid[i])) + c2_ext_grid[i] * np.log(n_ext_grid[i + 1] / n_ext_grid[i]))
            mu_high[i, :] = mu_ext_grid[i] * np.exp(slope * (n_high[i, :] - n_ext_grid[i] - n_ext_grid[i] * np.log(n_high[i, :] / n_ext_grid[i])) + c2_ext_grid[i] * np.log(n_high[i, :] / n_ext_grid[i]))

        self.n_high = n_high.flatten()
        self.c2_high = c2_high.flatten()
        self.mu_high = mu_high.flatten()

        self.e_high = cumulative_trapezoid(self.mu_high, self.n_high, initial=0) + self.e_at_n_connect
        self.p_high = self.mu_high * self.n_high - self.e_high

        n_low = self.n_low
        self.n_array = np.concatenate((self.n_low[n_low < self.n_connect], self.n_high))
        self.p_array = np.concatenate((self.p_low[n_low < self.n_connect], self.p_high))
        self.e_array = np.concatenate((self.e_low[n_low < self.n_connect], self.e_high))

    def __calculate_pseudo_enthalpy(self):

        intergrand = self.p_array / (self.e_array + self.p_array)
        self.h_array = cumulative_trapezoid(intergrand, np.log(self.p_array), initial=0) + intergrand[0]

    def __construct_all_interpolation(self):

        self.log_energy_density_from_log_pressure = interp1d(np.log(self.p_array),
                                                             np.log(self.e_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_energy_density_from_log_pseudo_enthalpy = interp1d(np.log(self.h_array),
                                                                    np.log(self.e_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)
        self.log_energy_density_from_log_number_density = interp1d(np.log(self.n_array),
                                                                   np.log(self.e_array), kind='linear',
                                                                   fill_value='extrapolate',
                                                                   assume_sorted=True)

        self.log_pressure_from_log_energy_density = interp1d(np.log(self.e_array),
                                                             np.log(self.p_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_pressure_from_log_number_density = interp1d(np.log(self.n_array),
                                                             np.log(self.p_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_pressure_from_log_pseudo_enthalpy = interp1d(np.log(self.h_array),
                                                              np.log(self.p_array), kind='linear',
                                                              fill_value='extrapolate',
                                                              assume_sorted=True)

        self.log_number_density_from_log_pressure = interp1d(np.log(self.p_array),
                                                             np.log(self.n_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_number_density_from_log_pseudo_enthalpy = interp1d(np.log(self.h_array),
                                                                    np.log(self.n_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)
        self.log_number_density_from_log_energy_density = interp1d(np.log(self.e_array),
                                                                   np.log(self.n_array), kind='linear',
                                                                   fill_value='extrapolate',
                                                                   assume_sorted=True)

        self.log_pseudo_enthalpy_from_log_pressure = interp1d(np.log(self.p_array),
                                                              np.log(self.h_array), kind='linear',
                                                              fill_value='extrapolate',
                                                              assume_sorted=True)
        self.log_pseudo_enthalpy_from_log_energy_density = interp1d(np.log(self.e_array),
                                                                    np.log(self.h_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)
        self.log_pseudo_enthalpy_from_log_number_density = interp1d(np.log(self.n_array),
                                                                    np.log(self.h_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)

        self.log_dedp_from_log_pressure = interp1d(np.log(self.p_array),
                                                   np.gradient(np.log(self.e_array), np.log(self.p_array)),
                                                   kind='linear',
                                                   fill_value='extrapolate',
                                                   assume_sorted=True)

    def energy_density_from_pressure(self, p):
        return np.exp(self.log_energy_density_from_log_pressure(np.log(p)))

    def energy_density_from_pseudo_enthalpy(self, h):
        return np.exp(self.log_energy_density_from_log_pseudo_enthalpy(np.log(h)))

    def energy_density_from_number_density(self, n):
        return np.exp(self.log_energy_density_from_log_number_density(np.log(n)))

    def pressure_from_energy_density(self, e):
        return np.exp(self.log_pressure_from_log_energy_density(np.log(e)))

    def pressure_from_pseudo_enthalpy(self, h):
        return np.exp(self.log_pressure_from_log_pseudo_enthalpy(np.log(h)))

    def pressure_from_number_density(self, n):
        return np.exp(self.log_pressure_from_log_number_density(np.log(n)))

    def number_density_from_pressure(self, p):
        return np.exp(self.log_number_density_from_log_pressure(np.log(p)))

    def number_density_from_pseudo_enthalpy(self, h):
        return np.exp(self.log_number_density_from_log_pseudo_enthalpy(np.log(h)))

    def number_density_from_energy_density(self, e):
        return np.exp(self.log_number_density_from_log_energy_density(np.log(e)))

    def pseudo_enthalpy_from_pressure(self, p):
        return np.exp(self.log_pseudo_enthalpy_from_log_pressure(np.log(p)))

    def pseudo_enthalpy_from_number_density(self, n):
        return np.exp(self.log_pseudo_enthalpy_from_log_number_density(np.log(n)))

    def pseudo_enthalpy_from_energy_density(self, e):
        return np.exp(self.log_pseudo_enthalpy_from_log_energy_density(np.log(e)))

    def dedp_from_pressure(self, p):
        e = self.energy_density_from_pressure(p)
        return e / p * self.log_dedp_from_log_pressure(np.log(p))

    def construct_family(self, ndat=100):

        pc_min = 3.5  # arbitary lower bound pc in MeV fm^-3
        pc_max = self.pressure_from_number_density(self.n_lim * 0.999)

        pcs = np.logspace(np.log10(pc_min), np.log10(pc_max), num=ndat)

        # Generate the arrays of mass, radius and k2
        ms = []
        rs = []
        ks = []
        logpcs = []

        for i, pc in enumerate(pcs):
            m, r, k2 = TOVSolver(self, pc)
            ms.append(m)
            rs.append(r)
            ks.append(k2)
            logpcs.append(np.log(pc))

            if len(ms) > 1 and ms[-1] < ms[-2]:
                break

        ms = np.array(ms)
        rs = np.array(rs)
        ks = np.array(ks)

        if i != ndat - 1:
            # build a interpolation for logpc-mass to get logpc at max mass
            f = interp1d(logpcs, -ms, kind='linear')
            res = minimize_scalar(f, method='bounded',
                                  bounds=(logpcs[0] * 1.001, logpcs[-1] * 0.999))
            logpmax = res.x
            pmax = np.exp(logpmax)
            mmax, rmax, kmax = TOVSolver(self, pmax)

            # replace the last entry with the actual maximum
            logpcs[-1] = logpmax
            ms[-1] = mmax
            rs[-1] = rmax
            ks[-1] = kmax

        # calculate the compactness
        cs = ms / rs

        # convert the mass to solar mass
        ms /= lal.MRSUN_SI
        # convert the radius to km
        rs /= 1e3

        # calculate the tidal deformability
        lambdas = 2. / 3. * ks * np.power(cs, -5.)

        # build the mass-lambda interpolation
        self.lambda_m_interp = interp1d(ms, lambdas, kind='linear')

        # build the mass-radius interpolation
        self.radius_m_interp = interp1d(ms, rs, kind='linear')

        return
