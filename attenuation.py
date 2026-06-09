from functools import lru_cache

import numpy as np


ALPHA_EM = 1.0 / 137.035999084
M_E_EV = 0.51099895e6
T_CMB_EV = 2.348e-4

M_PER_EV_INV = 1.973269804e-7
KPC_IN_M = 3.085677581e19
KPC_IN_EV_INV = KPC_IN_M / M_PER_EV_INV


def sigma_gg_epsc(epsc_eV):
    epsc = np.asarray(epsc_eV, dtype=float)

    x = np.maximum(epsc / M_E_EV, 1.0 + 1.0e-15)
    beta = np.sqrt(1.0 - 1.0 / x**2)

    pre = 0.5 * np.pi * ALPHA_EM**2 / M_E_EV**2
    one_minus_b2 = 1.0 - beta**2

    logterm = np.log((1.0 + beta) / (1.0 - beta))
    bracket = (3.0 - beta**4) * logterm - 2.0 * beta * (2.0 - beta**2)

    return pre * one_minus_b2 * bracket


def tau_cmb_scalar(E_GeV, s_kpc, n_u=800, u_max=1.0e3):
    E_eV = float(E_GeV) * 1.0e9
    s_eV_inv = float(s_kpc) * KPC_IN_EV_INV

    if E_eV <= 0.0 or s_eV_inv <= 0.0:
        return 0.0

    u = np.logspace(0.0, np.log10(u_max), int(n_u))
    epsc = u * M_E_EV

    sigma = sigma_gg_epsc(epsc)

    arg = epsc**2 / (E_eV * T_CMB_EV)
    ln_term = np.log1p(-np.exp(-np.clip(arg, 0.0, 700.0)))

    integrand = epsc**3 * sigma * ln_term
    integral = np.trapz(integrand, epsc)

    prefactor = 4.0 * T_CMB_EV * s_eV_inv / (np.pi**2 * E_eV**2)

    return -prefactor * integral


@lru_cache(maxsize=4096)
def tau_cmb_cached(E_GeV_rounded, s_kpc_rounded):
    return tau_cmb_scalar(E_GeV_rounded, s_kpc_rounded)


def tau_cmb(E_GeV, s_kpc, E_round=3, s_round=3):
    E, s = np.broadcast_arrays(
        np.asarray(E_GeV, dtype=float),
        np.asarray(s_kpc, dtype=float),
    )

    out = np.empty_like(E, dtype=float)

    it = np.nditer(
        [E, s, out],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["writeonly"]],
    )

    for Ei, si, oi in it:
        Er = round(float(Ei), E_round)
        sr = round(float(si), s_round)
        oi[...] = tau_cmb_cached(Er, sr)

    return out


def transmission_cmb(E_GeV, s_kpc):
    return np.exp(-tau_cmb(E_GeV, s_kpc))