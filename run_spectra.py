import matplotlib.pyplot as plt

from config import HDMSPECTRA_H5, FIGURE_DIR, require_file
from spectra import spectra_6nu


COL_TOTAL = "#4e4e4e"
COL_NUE = "limegreen"
COL_NUMU = "royalblue"
COL_NUTAU = "orangered"
COL_GAMMA = "mediumvioletred"


plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "axes.titlepad": 10,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
})


m_x = 1.0e9

h5_path = require_file(HDMSPECTRA_H5)

x, spec = spectra_6nu(
    m_x=m_x,
    h5_path=h5_path,
    n_z=96,
)

fig, ax = plt.subplots(figsize=(8.0, 5.2))

ax.plot(
    x,
    x**2 * spec["nu_total"],
    lw=3.5,
    color=COL_TOTAL,
    label=r"Total $\nu$ prompt spectrum",
)

ax.plot(
    x,
    x**2 * spec["nue"],
    lw=2.5,
    color=COL_NUE,
    label=r"$\nu_e$ prompt spectrum",
)

ax.plot(
    x,
    x**2 * spec["numu"],
    lw=2.5,
    color=COL_NUMU,
    label=r"$\nu_\mu$ prompt spectrum",
)

ax.plot(
    x,
    x**2 * spec["nutau"],
    lw=2.5,
    color=COL_NUTAU,
    label=r"$\nu_\tau$ prompt spectrum",
)

ax.plot(
    x,
    x**2 * spec["gamma"],
    lw=2.5,
    ls="--",
    color=COL_GAMMA,
    label=r"$\gamma$ prompt spectrum",
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1.0e-4, 1.0)
ax.set_ylim(1.0e-4, 2.0)

ax.set_xlabel(r"$x = E/m_\chi$")
ax.set_ylabel(r"$x^2\,\mathrm{d}N/\mathrm{d}x$")

ax.set_title(r"Prompt Spectrum")

ax.grid(True, which="major", ls="--", lw=0.7, alpha=0.5)
ax.grid(True, which="minor", ls="--", lw=0.5, alpha=0.3)

ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

leg = ax.legend(loc="upper left", frameon=True)
leg.get_frame().set_alpha(0.9)
leg.get_frame().set_linewidth(0.0)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "prompt_spectra_6nu.png", dpi=200)
plt.show()