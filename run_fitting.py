import matplotlib.pyplot as plt

from config import HDMSPECTRA_H5, FIGURE_DIR, require_file
from fluxes import km3_roi
from fitting import FluxPoint, fit_flux_point


COL_KM3 = "mediumvioletred"
COL_JOINT = "royalblue"
COL_MODEL = "#4e4e4e"


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


KM3NET_POINT = FluxPoint(
    name="KM3NeT",
    E_star=2.2e8,
    E2Phi=5.8e-8,
    one_sigma=(2.1e-8, 1.59e-7),
)


JOINT_POINT = FluxPoint(
    name="KM3NeT + IceCube",
    E_star=2.2e8,
    E2Phi=7.5e-10,
    one_sigma=((7.5 - 4.7) * 1.0e-10, (7.5 + 13.1) * 1.0e-10),
)


def setup_ax():
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1.0e6, 1.0e10)
    ax.set_ylim(1.0e-12, 1.0e-6)

    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(
        r"$E_\nu^2 \Phi_{\nu+\bar{\nu}}^{\rm single\ flavor}$ "
        r"[GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
    )

    ax.grid(True, which="major", ls="--", lw=0.7, alpha=0.5)
    ax.grid(True, which="minor", ls="--", lw=0.5, alpha=0.3)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    return fig, ax


def add_flux_point(ax, point, color):
    yerr_low = point.E2Phi - point.one_sigma[0]
    yerr_high = point.one_sigma[1] - point.E2Phi

    ax.errorbar(
        point.E_star,
        point.E2Phi,
        yerr=[[yerr_low], [yerr_high]],
        fmt="o",
        ms=7,
        color=color,
        capsize=4,
        label=point.name,
    )


def add_fit_curve(ax, result, color, label):
    E = result["grid"]["E"]
    Y = result["grid"]["E2Phi_best"]

    ax.plot(
        E,
        Y,
        lw=2.6,
        color=color,
        label=label,
    )


def print_result(result):
    tau_lo, tau_hi = result["tau_one_sigma"]

    print()
    print(result["name"])
    print("-" * len(result["name"]))
    print(f"m_X        = {result['m_x']:.4e} GeV")
    print(f"log10(m_X) = {result['log10_m_x']:.4f}")
    print(f"tau_X      = {result['tau_x']:.4e} s")
    print(f"tau_X 1σ   = [{tau_lo:.4e}, {tau_hi:.4e}] s")
    print(f"E_peak     = {result['E_peak']:.4e} GeV")
    print(f"E_target   = {result['E_star']:.4e} GeV")
    print(f"D_ROI      = {result['d_factor']:.4e} GeV cm^-2 sr^-1")


def main():
    h5_path = require_file(HDMSPECTRA_H5)

    roi = km3_roi(
        theta_deg=3.0,
        n_alpha=6,
        n_phi=12,
        n_s=80,
    )

    km3_fit = fit_flux_point(
        point=KM3NET_POINT,
        h5_path=h5_path,
        roi=roi,
        log10_m_bounds=(8.7, 12.5),
        n_z_spectra=48,
        n_z_eg=50,
        symmetric_x=True,
    )

    joint_fit = fit_flux_point(
        point=JOINT_POINT,
        h5_path=h5_path,
        roi=roi,
        log10_m_bounds=(8.7, 12.5),
        n_z_spectra=48,
        n_z_eg=50,
        symmetric_x=True,
    )

    print_result(km3_fit)
    print_result(joint_fit)

    fig, ax = setup_ax()

    add_flux_point(ax, KM3NET_POINT, COL_KM3)
    add_flux_point(ax, JOINT_POINT, COL_JOINT)

    add_fit_curve(
        ax,
        km3_fit,
        COL_KM3,
        rf"Best fit to KM3NeT: $m_X={km3_fit['m_x']:.2e}$ GeV",
    )

    add_fit_curve(
        ax,
        joint_fit,
        COL_JOINT,
        rf"Best fit to joint flux: $m_X={joint_fit['m_x']:.2e}$ GeV",
    )

    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "km3net_flux_fit.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()