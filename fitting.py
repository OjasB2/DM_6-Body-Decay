from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from spectra import (
    PDG_FINAL,
    democratic_6nu_primaries,
    spectrum_from_primaries,
)
from fluxes import neutrino_flux_component


NU_KEYS = ("nue", "numu", "nutau")


@dataclass
class FluxPoint:
    name: str
    E_star: float
    E2Phi: float
    one_sigma: tuple


def spectra_6nu_neutrinos(m_x, h5_path, x_grid, n_z_spectra=64):
    primaries = democratic_6nu_primaries()
    spectra = {}

    for key in NU_KEYS:
        spectra[key] = spectrum_from_primaries(
            x_grid=x_grid,
            final_pdg=PDG_FINAL[key],
            primaries=primaries,
            m_x=m_x,
            h5_path=h5_path,
            n_z=n_z_spectra,
        )

    return spectra


def single_flavor_flux(
    E_grid,
    spectra,
    x_grid,
    m_x,
    tau_x,
    d_factor,
    z_max=6.0,
    n_z_eg=60,
    symmetric_x=True,
):
    total_nu = np.zeros_like(E_grid, dtype=float)

    for key in NU_KEYS:
        comp = neutrino_flux_component(
            x_grid=x_grid,
            dndx=spectra[key],
            m_x=m_x,
            tau_x=tau_x,
            d_factor=d_factor,
            z_max=z_max,
            n_z=n_z_eg,
        )
        total_nu += comp["total"]

    if symmetric_x:
        return (2.0 / 3.0) * total_nu

    return (1.0 / 3.0) * total_nu


def e2phi_single_flavor(
    E_grid,
    m_x,
    tau_x,
    h5_path,
    d_factor,
    n_z_spectra=64,
    n_z_eg=60,
    z_max=6.0,
    symmetric_x=True,
):
    E_grid = np.asarray(E_grid, dtype=float)
    x_grid = E_grid / m_x

    spectra = spectra_6nu_neutrinos(
        m_x=m_x,
        h5_path=h5_path,
        x_grid=x_grid,
        n_z_spectra=n_z_spectra,
    )

    phi = single_flavor_flux(
        E_grid=E_grid,
        spectra=spectra,
        x_grid=x_grid,
        m_x=m_x,
        tau_x=tau_x,
        d_factor=d_factor,
        z_max=z_max,
        n_z_eg=n_z_eg,
        symmetric_x=symmetric_x,
    )

    return E_grid**2 * phi


def safe_log(x, eps=1.0e-300):
    return np.log(np.clip(x, eps, None))


def parabolic_peak(E, Y, i):
    if i <= 0 or i >= len(E) - 1:
        return E[i], Y[i]

    x = np.log(E[i - 1:i + 2])
    y = np.log(np.clip(Y[i - 1:i + 2], 1.0e-300, None))

    a, b, c = np.polyfit(x, y, 2)

    if a >= 0.0:
        return E[i], Y[i]

    x_peak = -b / (2.0 * a)
    y_peak = a * x_peak**2 + b * x_peak + c

    return np.exp(x_peak), np.exp(y_peak)


def peak_near_energy(
    m_x,
    E_star,
    h5_path,
    d_factor,
    span=2.5,
    n_E=100,
    max_expand=3,
    expand_factor=1.8,
    n_z_spectra=64,
    n_z_eg=60,
    z_max=6.0,
    symmetric_x=True,
):
    for _ in range(max_expand + 1):
        E_min = max(E_star / span, 1.0e-6 * m_x)
        E_max = min(E_star * span, 0.5 * m_x)

        if E_max <= E_min:
            span *= expand_factor
            continue

        E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_E)

        Y = e2phi_single_flavor(
            E_grid=E_grid,
            m_x=m_x,
            tau_x=1.0,
            h5_path=h5_path,
            d_factor=d_factor,
            n_z_spectra=n_z_spectra,
            n_z_eg=n_z_eg,
            z_max=z_max,
            symmetric_x=symmetric_x,
        )

        if not np.any(np.isfinite(Y)) or np.all(Y <= 0.0):
            span *= expand_factor
            continue

        i = int(np.nanargmax(Y))

        if 0 < i < len(E_grid) - 1:
            E_peak, Y_peak = parabolic_peak(E_grid, Y, i)
            return E_peak, Y_peak, E_grid, Y

        span *= expand_factor

    i = int(np.nanargmax(Y))
    return E_grid[i], Y[i], E_grid, Y


def solve_mass_from_peak(
    E_star,
    h5_path,
    d_factor,
    log10_m_bounds=(8.7, 12.5),
    n_scan=10,
    **kwargs,
):
    cache = {}

    def objective(log10_m):
        key = round(float(log10_m), 6)

        if key not in cache:
            m_x = 10.0**log10_m
            E_peak, Y_peak, E_grid, Y = peak_near_energy(
                m_x=m_x,
                E_star=E_star,
                h5_path=h5_path,
                d_factor=d_factor,
                **kwargs,
            )
            cache[key] = {
                "m_x": m_x,
                "E_peak": E_peak,
                "Y_peak": Y_peak,
                "E_grid": E_grid,
                "Y": Y,
            }

        E_peak = cache[key]["E_peak"]

        if not np.isfinite(E_peak) or E_peak <= 0.0:
            return np.inf

        return np.log(E_peak / E_star)

    lo, hi = log10_m_bounds
    grid = np.linspace(lo, hi, n_scan)
    vals = np.array([objective(g) for g in grid])

    bracket = None

    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0.0:
            bracket = (a, b)
            break

    if bracket is not None:
        log10_m_best = brentq(objective, bracket[0], bracket[1], xtol=1.0e-3)
    else:
        result = minimize_scalar(
            lambda x: abs(objective(x)),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1.0e-3},
        )
        log10_m_best = float(result.x)

    m_best = 10.0**log10_m_best

    E_peak, Y_peak, E_grid, Y = peak_near_energy(
        m_x=m_best,
        E_star=E_star,
        h5_path=h5_path,
        d_factor=d_factor,
        **kwargs,
    )

    return m_best, E_peak, Y_peak, E_grid, Y


def fit_flux_point(
    point,
    h5_path,
    roi,
    log10_m_bounds=(8.7, 12.5),
    n_z_spectra=64,
    n_z_eg=60,
    z_max=6.0,
    symmetric_x=True,
):
    d_factor = roi.d_factor()

    m_best, E_peak, Y_peak, E_grid, Y_tau1 = solve_mass_from_peak(
        E_star=point.E_star,
        h5_path=h5_path,
        d_factor=d_factor,
        log10_m_bounds=log10_m_bounds,
        n_z_spectra=n_z_spectra,
        n_z_eg=n_z_eg,
        z_max=z_max,
        symmetric_x=symmetric_x,
    )

    Y_star_tau1 = e2phi_single_flavor(
        E_grid=np.array([point.E_star]),
        m_x=m_best,
        tau_x=1.0,
        h5_path=h5_path,
        d_factor=d_factor,
        n_z_spectra=n_z_spectra,
        n_z_eg=n_z_eg,
        z_max=z_max,
        symmetric_x=symmetric_x,
    )[0]

    tau_best = Y_star_tau1 / point.E2Phi

    low, high = point.one_sigma
    tau_low = Y_star_tau1 / high
    tau_high = Y_star_tau1 / low

    return {
        "name": point.name,
        "m_x": m_best,
        "log10_m_x": np.log10(m_best),
        "tau_x": tau_best,
        "tau_one_sigma": (tau_low, tau_high),
        "E_star": point.E_star,
        "E_peak": E_peak,
        "E2Phi_target": point.E2Phi,
        "Y_star_tau1": Y_star_tau1,
        "d_factor": d_factor,
        "grid": {
            "E": E_grid,
            "E2Phi_tau1": Y_tau1,
            "E2Phi_best": Y_tau1 / tau_best,
        },
    }