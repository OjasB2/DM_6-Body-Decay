from dataclasses import dataclass

import numpy as np


C_LIGHT = 2.99792458e10
KPC_TO_CM = 3.085677581e21
MPC_TO_CM = 3.085677581e24

H0_KM_S_MPC = 67.4
H0 = H0_KM_S_MPC * 1.0e5 / MPC_TO_CM
OMEGA_M = 0.315
OMEGA_L = 0.685

RHO_DM_AVG = 1.15e-6

R_MW = 40.0
R_SUN = 8.0
R_S = 24.0
RHO_S = 0.18

NU_KEYS = ("nue", "numu", "nutau")


@dataclass
class ConeROI:
    name: str
    l0: float
    b0: float
    theta: float
    n_alpha: int = 10
    n_phi: int = 20
    n_s: int = 120

    def d_factor(self):
        return d_factor_cone(
            l0=self.l0,
            b0=self.b0,
            theta=self.theta,
            n_alpha=self.n_alpha,
            n_phi=self.n_phi,
            n_s=self.n_s,
        )

    def d_factor_attenuated(self, E_grid, transmission):
        return d_factor_cone(
            l0=self.l0,
            b0=self.b0,
            theta=self.theta,
            n_alpha=self.n_alpha,
            n_phi=self.n_phi,
            n_s=self.n_s,
            E_grid=E_grid,
            transmission=transmission,
        )


@dataclass
class RectROI:
    name: str
    l1: float
    l2: float
    b1: float
    b2: float
    n_l: int = 30
    n_b: int = 14
    n_s: int = 120

    def d_factor(self):
        return d_factor_rect(
            l1=self.l1,
            l2=self.l2,
            b1=self.b1,
            b2=self.b2,
            n_l=self.n_l,
            n_b=self.n_b,
            n_s=self.n_s,
        )

    def d_factor_attenuated(self, E_grid, transmission):
        return d_factor_rect(
            l1=self.l1,
            l2=self.l2,
            b1=self.b1,
            b2=self.b2,
            n_l=self.n_l,
            n_b=self.n_b,
            n_s=self.n_s,
            E_grid=E_grid,
            transmission=transmission,
        )


def hubble(z):
    return H0 * np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def rho_nfw(r_kpc):
    x = r_kpc / R_S
    return RHO_S / (x * (1.0 + x) ** 2)


def galactic_radius(s_kpc, l_rad, b_rad):
    cos_psi = np.cos(l_rad) * np.cos(b_rad)
    return np.sqrt(s_kpc**2 + R_SUN**2 - 2.0 * s_kpc * R_SUN * cos_psi)


def los_upper_limit(l_rad, b_rad):
    cos_psi = np.cos(l_rad) * np.cos(b_rad)
    sin2_psi = 1.0 - cos_psi**2
    return np.sqrt(R_MW**2 - R_SUN**2 * sin2_psi) + R_SUN * cos_psi


def los_integral(l_rad, b_rad, n_s=120, E_grid=None, transmission=None):
    s_max = los_upper_limit(l_rad, b_rad)

    if s_max <= 0.0:
        if E_grid is None:
            return 0.0
        return np.zeros_like(np.asarray(E_grid, dtype=float))

    xs, ws = np.polynomial.legendre.leggauss(n_s)

    s = 0.5 * (xs + 1.0) * s_max
    w = 0.5 * s_max * ws

    rho = rho_nfw(galactic_radius(s, l_rad, b_rad))

    if E_grid is None or transmission is None:
        return np.sum(w * rho) * KPC_TO_CM

    E = np.asarray(E_grid, dtype=float)

    att = transmission(E[None, :], s[:, None])
    val = np.sum(w[:, None] * rho[:, None] * att, axis=0)

    return val * KPC_TO_CM


def d_factor_rect(
    l1,
    l2,
    b1,
    b2,
    n_l=30,
    n_b=14,
    n_s=120,
    E_grid=None,
    transmission=None,
):
    delta_omega = (l2 - l1) * (np.sin(b2) - np.sin(b1))

    if delta_omega <= 0.0:
        raise ValueError("ROI has non-positive solid angle.")

    xl, wl = np.polynomial.legendre.leggauss(n_l)
    xb, wb = np.polynomial.legendre.leggauss(n_b)

    l_grid = 0.5 * (xl + 1.0) * (l2 - l1) + l1
    b_grid = 0.5 * (xb + 1.0) * (b2 - b1) + b1

    dl = 0.5 * (l2 - l1)
    db = 0.5 * (b2 - b1)

    if E_grid is None or transmission is None:
        total = 0.0
    else:
        total = np.zeros_like(np.asarray(E_grid, dtype=float))

    for i, l in enumerate(l_grid):
        for j, b in enumerate(b_grid):
            weight = wl[i] * dl * wb[j] * db * np.cos(b)
            total += weight * los_integral(
                l_rad=l,
                b_rad=b,
                n_s=n_s,
                E_grid=E_grid,
                transmission=transmission,
            )

    return total / delta_omega


def d_factor_cone(
    l0,
    b0,
    theta,
    n_alpha=10,
    n_phi=20,
    n_s=120,
    E_grid=None,
    transmission=None,
):
    delta_omega = 2.0 * np.pi * (1.0 - np.cos(theta))

    if delta_omega <= 0.0:
        raise ValueError("ROI has non-positive solid angle.")

    xa, wa = np.polynomial.legendre.leggauss(n_alpha)

    alpha_grid = 0.5 * (xa + 1.0) * theta
    dalpha = 0.5 * theta

    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    if E_grid is None or transmission is None:
        total = 0.0
    else:
        total = np.zeros_like(np.asarray(E_grid, dtype=float))

    for i, alpha in enumerate(alpha_grid):
        for phi in phi_grid:
            l, b = rotate_from_center(l0, b0, alpha, phi)
            weight = wa[i] * dalpha * dphi * np.sin(alpha)

            total += weight * los_integral(
                l_rad=l,
                b_rad=b,
                n_s=n_s,
                E_grid=E_grid,
                transmission=transmission,
            )

    return total / delta_omega


def rotate_from_center(l0, b0, alpha, phi):
    cl, sl = np.cos(l0), np.sin(l0)
    cb, sb = np.cos(b0), np.sin(b0)

    n = np.array([cb * cl, cb * sl, sb])
    e_theta = np.array([sb * cl, sb * sl, -cb])
    e_phi = np.array([-sl, cl, 0.0])

    v = np.cos(alpha) * n + np.sin(alpha) * (
        np.cos(phi) * e_theta + np.sin(phi) * e_phi
    )

    l = np.arctan2(v[1], v[0]) % (2.0 * np.pi)
    b = np.arcsin(v[2])

    return l, b


def equatorial_to_galactic(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    v_eq = np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])

    rot = np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [ 0.4941094279, -0.4448296300,  0.7469822445],
        [-0.8676661490, -0.1980763734,  0.4559837762],
    ])

    v_gal = rot @ v_eq

    l = np.arctan2(v_gal[1], v_gal[0]) % (2.0 * np.pi)
    b = np.arcsin(v_gal[2])

    return l, b


def dndx_to_dnde(x_grid, dndx, m_x):
    E = m_x * np.asarray(x_grid, dtype=float)
    dnde = np.asarray(dndx, dtype=float) / m_x
    return E, dnde


def spectrum_interpolator(E_grid, dnde):
    E_grid = np.asarray(E_grid, dtype=float)
    dnde = np.asarray(dnde, dtype=float)

    mask = (E_grid > 0.0) & (dnde > 0.0)

    if np.sum(mask) < 2:
        return lambda E: np.zeros_like(np.asarray(E, dtype=float))

    logE = np.log(E_grid[mask])
    logY = np.log(dnde[mask])

    def interp(E):
        E = np.asarray(E, dtype=float)
        out = np.zeros_like(E)

        good = E > 0.0
        y = np.interp(
            np.log(E[good]),
            logE,
            logY,
            left=-np.inf,
            right=-np.inf,
        )

        out[good] = np.exp(y)
        return out

    return interp


def galactic_flux(E_grid, dnde, m_x, tau_x, d_factor):
    prefactor = 1.0 / (4.0 * np.pi * m_x * tau_x)
    return prefactor * d_factor * dnde


def extragalactic_neutrino_flux(E_grid, dnde, m_x, tau_x, z_max=6.0, n_z=80):
    E = np.asarray(E_grid, dtype=float)

    z_nodes, z_weights = np.polynomial.legendre.leggauss(n_z)
    z = 0.5 * (z_nodes + 1.0) * z_max
    w = 0.5 * z_max * z_weights

    spec = spectrum_interpolator(E_grid, dnde)

    E_emit = (1.0 + z[:, None]) * E[None, :]
    integrand = spec(E_emit) / hubble(z)[:, None]

    integral = np.sum(w[:, None] * integrand, axis=0)

    prefactor = C_LIGHT * RHO_DM_AVG / (4.0 * np.pi * m_x * tau_x)
    return prefactor * integral


def neutrino_flux_component(x_grid, dndx, m_x, tau_x, d_factor, z_max=6.0, n_z=80):
    E, dnde = dndx_to_dnde(x_grid, dndx, m_x)

    gal = galactic_flux(
        E_grid=E,
        dnde=dnde,
        m_x=m_x,
        tau_x=tau_x,
        d_factor=d_factor,
    )

    eg = extragalactic_neutrino_flux(
        E_grid=E,
        dnde=dnde,
        m_x=m_x,
        tau_x=tau_x,
        z_max=z_max,
        n_z=n_z,
    )

    return {
        "E": E,
        "gal": gal,
        "eg": eg,
        "total": gal + eg,
    }


def gamma_flux_component(x_grid, dndx, m_x, tau_x, roi, d_factor, transmission=None):
    E, dnde = dndx_to_dnde(x_grid, dndx, m_x)

    gal_noatt = galactic_flux(
        E_grid=E,
        dnde=dnde,
        m_x=m_x,
        tau_x=tau_x,
        d_factor=d_factor,
    )

    if transmission is None:
        d_factor_att = d_factor
        gal_att = gal_noatt.copy()
    else:
        d_factor_att = roi.d_factor_attenuated(
            E_grid=E,
            transmission=transmission,
        )

        gal_att = galactic_flux(
            E_grid=E,
            dnde=dnde,
            m_x=m_x,
            tau_x=tau_x,
            d_factor=d_factor_att,
        )

    return {
        "E": E,
        "gal_noatt": gal_noatt,
        "gal_att": gal_att,
        "d_factor_noatt": d_factor,
        "d_factor_att": d_factor_att,
    }


def compute_fluxes(
    x_grid,
    spectra,
    m_x,
    tau_x,
    roi,
    z_max=6.0,
    n_z=80,
    photon_transmission=None,
):
    out = {}

    d_factor = roi.d_factor()

    for key in NU_KEYS:
        out[key] = neutrino_flux_component(
            x_grid=x_grid,
            dndx=spectra[key],
            m_x=m_x,
            tau_x=tau_x,
            d_factor=d_factor,
            z_max=z_max,
            n_z=n_z,
        )

    E = out["nue"]["E"]

    nu_gal = sum(out[key]["gal"] for key in NU_KEYS)
    nu_eg = sum(out[key]["eg"] for key in NU_KEYS)

    out["nu_total"] = {
        "E": E,
        "gal": nu_gal,
        "eg": nu_eg,
        "total": nu_gal + nu_eg,
    }

    out["gamma"] = gamma_flux_component(
        x_grid=x_grid,
        dndx=spectra["gamma"],
        m_x=m_x,
        tau_x=tau_x,
        roi=roi,
        d_factor=d_factor,
        transmission=photon_transmission,
    )

    out["roi"] = roi
    out["d_factor"] = d_factor

    return out


def km3_roi(theta_deg=3.0, n_alpha=10, n_phi=20, n_s=120):
    l0, b0 = equatorial_to_galactic(ra_deg=94.3, dec_deg=-7.8)

    return ConeROI(
        name=rf"KM3NeT cone, $\theta={theta_deg:g}^\circ$",
        l0=l0,
        b0=b0,
        theta=np.deg2rad(theta_deg),
        n_alpha=n_alpha,
        n_phi=n_phi,
        n_s=n_s,
    )


def lhaaso_inner_gp_roi(n_l=30, n_b=14, n_s=120):
    return RectROI(
        name=r"LHAASO inner Galactic plane",
        l1=np.deg2rad(15.0),
        l2=np.deg2rad(125.0),
        b1=np.deg2rad(-5.0),
        b2=np.deg2rad(5.0),
        n_l=n_l,
        n_b=n_b,
        n_s=n_s,
    )


def lhaaso_outer_gp_roi(n_l=30, n_b=14, n_s=120):
    return RectROI(
        name=r"LHAASO outer Galactic plane",
        l1=np.deg2rad(125.0),
        l2=np.deg2rad(235.0),
        b1=np.deg2rad(-5.0),
        b2=np.deg2rad(5.0),
        n_l=n_l,
        n_b=n_b,
        n_s=n_s,
    )