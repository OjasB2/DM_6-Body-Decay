import matplotlib.pyplot as plt

from config import HDMSPECTRA_H5, FIGURE_DIR, require_file
from spectra import spectra_6nu
from fluxes import compute_fluxes, lhaaso_outer_gp_roi
from attenuation import transmission_cmb


COL_TOTAL = "#4e4e4e"
COL_GAL = "royalblue"
COL_EG = "mediumvioletred"
COL_ATT = "limegreen"


plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
})


def setup_ax(title, ylabel, xlim=(1.0e4, 1.0e10), ylim=(1.0e-14, 1.0e-6)):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, which="major", ls="--", lw=0.7, alpha=0.5)
    ax.grid(True, which="minor", ls="--", lw=0.5, alpha=0.3)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    return fig, ax


def plot_neutrino_flux(out):
    E = out["nu_total"]["E"]

    fig, ax = setup_ax(
        title=r"Neutrino Flux",
        ylabel=r"$E_\nu^2\Phi_\nu$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        xlim=(1.0e4, 1.0e10),
        ylim=(1.0e-14, 1.0e-6),
    )

    ax.plot(
        E,
        E**2 * out["nu_total"]["total"],
        lw=3.0,
        color=COL_TOTAL,
        label=r"Total $\nu$",
    )

    ax.plot(
        E,
        E**2 * out["nu_total"]["gal"],
        lw=2.2,
        color=COL_GAL,
        label=r"Galactic $\nu$",
    )

    ax.plot(
        E,
        E**2 * out["nu_total"]["eg"],
        lw=2.2,
        color=COL_EG,
        label=r"Extragalactic $\nu$",
    )

    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "flux_neutrino.png", dpi=200)

    return fig, ax


def plot_photon_flux(out):
    E = out["gamma"]["E"]

    fig, ax = setup_ax(
        title=r"Galactic Photon Flux",
        ylabel=r"$E_\gamma^2\Phi_\gamma$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
        xlim=(1.0e4, 1.0e10),
        ylim=(1.0e-14, 1.0e-6),
    )

    ax.plot(
        E,
        E**2 * out["gamma"]["gal_noatt"],
        lw=2.8,
        color=COL_TOTAL,
        label=r"Unattenuated",
    )

    ax.plot(
        E,
        E**2 * out["gamma"]["gal_att"],
        lw=2.8,
        ls="--",
        color=COL_ATT,
        label=r"CMB attenuated",
    )

    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "flux_photon_cmb_attenuated.png", dpi=200)

    return fig, ax


def main():
    m_x = 1.0e9
    tau_x = 5.0e29

    h5_path = require_file(HDMSPECTRA_H5)

    x, spectra = spectra_6nu(
        m_x=m_x,
        h5_path=h5_path,
        n_z=96,
    )

    roi = lhaaso_outer_gp_roi(
        n_l=24,
        n_b=12,
        n_s=100,
    )

    out = compute_fluxes(
        x_grid=x,
        spectra=spectra,
        m_x=m_x,
        tau_x=tau_x,
        roi=roi,
        z_max=6.0,
        n_z=80,
        photon_transmission=None,
        #photon_transmission=transmission_cmb, for attenuation
    )

    print(f"ROI: {roi.name}")
    print(f"D_ROI = {out['d_factor']:.4e} GeV cm^-2 sr^-1")
    print(f"m_X = {m_x:.3e} GeV")
    print(f"tau_X = {tau_x:.3e} s")

    plot_neutrino_flux(out)
    plot_photon_flux(out)

    plt.show()


if __name__ == "__main__":
    main()