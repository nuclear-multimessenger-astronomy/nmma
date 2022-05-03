import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants


# unit conversion for pressure / energy_density
particle_to_SI = scipy.constants.e * 1e51
SI_to_geometric = scipy.constants.G / np.power(scipy.constants.c, 4.0)
particle_to_geometric = particle_to_SI * SI_to_geometric


def tov_ode(h, y, eos):
    r, m, H, b = y
    e = eos.energy_density_from_pseudo_enthalpy(h) * particle_to_geometric
    p = eos.pressure_from_pseudo_enthalpy(h) * particle_to_geometric
    dedp = e / p * eos.log_dedp_from_log_pressure(np.log(p / particle_to_geometric))

    A = 1.0 / (1.0 - 2.0 * m / r)
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * np.pi * r * (p - e))
    C0 = A * (
        -(2) * (2 + 1) / (r * r)
        + 4.0 * np.pi * (e + p) * dedp
        + 4.0 * np.pi * (5.0 * e + 9.0 * p)
    ) - np.power(2.0 * (m + 4.0 * np.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)

    drdh = -r * (r - 2.0 * m) / (m + 4.0 * np.pi * r * r * r * p)
    dmdh = 4.0 * np.pi * r * r * e * drdh
    dHdh = b * drdh
    dbdh = -(C0 * H + C1 * b) * drdh

    dydt = [drdh, dmdh, dHdh, dbdh]

    return dydt


def calc_k2(R, M, H, b):

    y = R * b / H
    C = M / R

    num = (
        (8.0 / 5.0)
        * np.power(1 - 2 * C, 2.0)
        * np.power(C, 5.0)
        * (2 * C * (y - 1) - y + 2)
    )
    den = (
        2
        * C
        * (
            4 * (y + 1) * np.power(C, 4)
            + (6 * y - 4) * np.power(C, 3)
            + (26 - 22 * y) * C * C
            + 3 * (5 * y - 8) * C
            - 3 * y
            + 6
        )
    )
    den -= (
        3
        * np.power(1 - 2 * C, 2)
        * (2 * C * (y - 1) - y + 2)
        * np.log(1.0 / (1 - 2 * C))
    )

    return num / den


def TOVSolver(eos, pc_pp):

    # central values
    hc = eos.pseudo_enthalpy_from_pressure(pc_pp)
    ec = eos.energy_density_from_pressure(pc_pp) * particle_to_geometric
    pc = pc_pp * particle_to_geometric
    dedp_c = eos.dedp_from_pressure(pc_pp)
    dhdp_c = 1.0 / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    # initial values
    dh = -1e-3 * hc
    h0 = hc + dh
    h1 = -dh
    r0 = np.sqrt(3.0 * (-dh) / 2.0 / np.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * np.pi * ec * np.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    H0 = r0 * r0
    b0 = 2.0 * r0

    y0 = [r0, m0, H0, b0]

    sol = solve_ivp(tov_ode, (h0, h1), y0, args=(eos,), rtol=1e-3, atol=0.0)

    # take one final Euler step to get to the surface
    R = sol.y[0, -1]
    M = sol.y[1, -1]
    H = sol.y[2, -1]
    b = sol.y[3, -1]

    y1 = [R, M, H, b]
    dydt1 = tov_ode(h1, y1, eos)

    for i in range(0, len(y1)):
        y1[i] += dydt1[i] * (0.0 - h1)

    R, M, H, b = y1
    k2 = calc_k2(R, M, H, b)

    return M, R, k2
