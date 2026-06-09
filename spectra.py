import numpy as np
from numpy.polynomial.legendre import leggauss
from HDMSpectra.HDMSpectra import FF


PDG_NU = {
    "e": 12,
    "mu": 14,
    "tau": 16,
}

PDG_FINAL = {
    "nue": 12,
    "numu": 14,
    "nutau": 16,
    "gamma": 22,
}


def gauss_nodes(a, b, n):
    x, w = leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def normalized_dgamma_dz(z):
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)

    mask = (z >= 0.0) & (z <= 0.5)
    zz = z[mask]

    out[mask] = 1344.0 * zz**2 * (1.0 - 2.0 * zz) ** 5
    return out


def ff_delta_coeff(final_pdg, primary_pdg, q, h5_path):
    val = FF(
        id_f=final_pdg,
        id_i=primary_pdg,
        xvals=np.array([1.0e-3]),
        Qval=q,
        data=str(h5_path),
        delta=True,
        interpolation="linear",
    )
    return float(val[-1])


def convolve_primary(
    x_grid,
    final_pdg,
    primary_pdg,
    multiplicity,
    m_x,
    h5_path,
    n_z=96,
    q_min=500.0,
):
    x = np.asarray(x_grid, dtype=float)
    dndx = np.zeros_like(x)

    z_min = max(q_min / m_x, 0.0)
    z_max = 0.5

    if z_min >= z_max:
        return dndx

    z_nodes, z_weights = gauss_nodes(z_min, z_max, n_z)
    pdf_nodes = normalized_dgamma_dz(z_nodes)

    for z, wz, pdf in zip(z_nodes, z_weights, pdf_nodes):
        mask = (x > 0.0) & (x <= z)
        if not np.any(mask):
            continue

        x_ff = np.clip(x[mask] / z, 1.0e-6, 1.0 - 1.0e-9)
        q = z * m_x

        ff = FF(
            id_f=final_pdg,
            id_i=primary_pdg,
            xvals=x_ff,
            Qval=q,
            data=str(h5_path),
            delta=False,
            interpolation="linear",
        )

        dndx[mask] += wz * pdf * ff / z

    if final_pdg != 22:
        line_mask = (x > z_min) & (x <= z_max)
        x_line = x[line_mask]

        if len(x_line) > 0:
            pdf_line = normalized_dgamma_dz(x_line)

            delta = np.array([
                ff_delta_coeff(final_pdg, primary_pdg, q=x_i * m_x, h5_path=h5_path)
                for x_i in x_line
            ])

            dndx[line_mask] += pdf_line * delta / x_line

    return multiplicity * dndx


def democratic_6nu_primaries():
    return [
        {"pdg": PDG_NU["e"], "multiplicity": 2.0},
        {"pdg": PDG_NU["mu"], "multiplicity": 2.0},
        {"pdg": PDG_NU["tau"], "multiplicity": 2.0},
    ]


def spectrum_from_primaries(
    x_grid,
    final_pdg,
    primaries,
    m_x,
    h5_path,
    n_z=96,
):
    total = np.zeros_like(np.asarray(x_grid, dtype=float))

    for primary in primaries:
        total += convolve_primary(
            x_grid=x_grid,
            final_pdg=final_pdg,
            primary_pdg=primary["pdg"],
            multiplicity=primary["multiplicity"],
            m_x=m_x,
            h5_path=h5_path,
            n_z=n_z,
        )

    return total


def spectra_6nu(m_x, h5_path, x_grid=None, n_z=96):
    if x_grid is None:
        x_grid = np.logspace(-6, np.log10(0.5), 500)

    primaries = democratic_6nu_primaries()

    spectra = {}

    for name, pdg in PDG_FINAL.items():
        spectra[name] = spectrum_from_primaries(
            x_grid=x_grid,
            final_pdg=pdg,
            primaries=primaries,
            m_x=m_x,
            h5_path=h5_path,
            n_z=n_z,
        )

    spectra["nu_total"] = spectra["nue"] + spectra["numu"] + spectra["nutau"]

    return x_grid, spectra


def dndx_to_dnde(dndx, m_x):
    return np.asarray(dndx) / m_x