import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.style as style
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import pandas as pd
import seaborn as sns

from astropy.time import Time


nterms = 3
sns.set_style("ticks")
sns.set_context("paper")

style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 21}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 54})
rc('ytick', **{'labelsize': 54})
rc('axes', **{'labelsize': 48, 'titlesize': 63})

fig, axs = plt.subplots(figsize=(28, 16), dpi=600, nrows=3, ncols=1, sharex=True,
                        gridspec_kw=dict(wspace=0.0, hspace=0.0))

data_res_full = pd.read_table("./NANOGrav_12yv4/narrowband/resid/res_full/J1643-1224_NANOGrav_12yv4.all.res",
                              delim_whitespace=True, comment='#')
data_res_avg = pd.read_table("./NANOGrav_12yv4/narrowband/resid/res_avg/J1643-1224_NANOGrav_12yv4.avg.res",
                             delim_whitespace=True, comment='#')

# Plot 1

Rcvr_800_GASP = np.where(data_res_full["flag"] == "Rcvr_800_GASP")[0]
Rcvr_1_2_GASP = np.where(data_res_full["flag"] == "Rcvr1_2_GASP")[0]
Rcvr_800_GUPPI = np.where(data_res_full["flag"] == "Rcvr_800_GUPPI")[0]
Rcvr1_2_GUPPI = np.where(data_res_full["flag"] == "Rcvr1_2_GUPPI")[0]

axs[0].errorbar(data_res_full["MJD"][Rcvr_800_GASP],
                data_res_full["residual(us)"][Rcvr_800_GASP],
                yerr=data_res_full["uncertainty(us)"][Rcvr_800_GASP],
                fmt='v', elinewidth=4, capthick=4, alpha=0.3,
                markersize=10, markeredgewidth=3,
#                c="#0FA3B1",
                label="Rcvr_800_GASP",  rasterized=True)

axs[0].errorbar(data_res_full["MJD"][Rcvr_1_2_GASP],
                data_res_full["residual(us)"][Rcvr_1_2_GASP],
                yerr=data_res_full["uncertainty(us)"][Rcvr_1_2_GASP],
                fmt='^', elinewidth=4, capthick=4, alpha=0.3,
                markersize=10, markeredgewidth=3,
#                c="#8CAF83",
                label="Rcvr_1_2_GASP",  rasterized=True)

axs[0].errorbar(data_res_full["MJD"][Rcvr_800_GUPPI],
                data_res_full["residual(us)"][Rcvr_800_GUPPI],
                yerr=data_res_full["uncertainty(us)"][Rcvr_800_GUPPI],
                fmt='v', elinewidth=4, capthick=4, alpha=0.3,
                markersize=10, markeredgewidth=3,
#                c="#A083AF",
                label="Rcvr_800_GUPPI",  rasterized=True)

axs[0].errorbar(data_res_full["MJD"][Rcvr1_2_GUPPI],
                data_res_full["residual(us)"][Rcvr1_2_GUPPI],
                yerr=data_res_full["uncertainty(us)"][Rcvr1_2_GUPPI],
                fmt='^', elinewidth=4, capthick=4, alpha=0.3,
                markersize=10, markeredgewidth=3,
#                c="#A73955",
                label="Rcvr1_2_GUPPI",  rasterized=True)

axs[0].axhline(0.0, color='grey', ls="--", lw=3, zorder=0)
axs[0].axvline(Time(2010.2, format='decimalyear', scale='utc').mjd, color='black', ls="--", lw=4, zorder=0)

# Plot 2 and 3

Rcvr_800_GASP = np.where(data_res_avg["flag"] == "Rcvr_800_GASP")[0]
Rcvr_1_2_GASP = np.where(data_res_avg["flag"] == "Rcvr1_2_GASP")[0]
Rcvr_800_GUPPI = np.where(data_res_avg["flag"] == "Rcvr_800_GUPPI")[0]
Rcvr1_2_GUPPI = np.where(data_res_avg["flag"] == "Rcvr1_2_GUPPI")[0]

for ax in fig.axes[1:]:
    ax.errorbar(data_res_avg["MJD"][Rcvr_800_GASP],
                data_res_avg["residual(us)"][Rcvr_800_GASP],
                yerr=data_res_avg["uncertainty(us)"][Rcvr_800_GASP],
                fmt='v', elinewidth=4, capthick=4,
                markersize=10, markeredgewidth=3,
#                c="#0FA3B1",
                label="GASP Rcvr_800")

    ax.errorbar(data_res_avg["MJD"][Rcvr_1_2_GASP],
                data_res_avg["residual(us)"][Rcvr_1_2_GASP],
                yerr=data_res_avg["uncertainty(us)"][Rcvr_1_2_GASP],
                fmt='^', elinewidth=4, capthick=4,
                markersize=10, markeredgewidth=3,
#                c="#8CAF83",
                label="GASP Rcvr_1_2")

    ax.errorbar(data_res_avg["MJD"][Rcvr_800_GUPPI],
                data_res_avg["residual(us)"][Rcvr_800_GUPPI],
                yerr=data_res_avg["uncertainty(us)"][Rcvr_800_GUPPI],
                fmt='v', elinewidth=4, capthick=4,
                markersize=10, markeredgewidth=3,
#                c="#A083AF",
                label="GUPPI Rcvr_800")

    ax.errorbar(data_res_avg["MJD"][Rcvr1_2_GUPPI],
                data_res_avg["residual(us)"][Rcvr1_2_GUPPI],
                yerr=data_res_avg["uncertainty(us)"][Rcvr1_2_GUPPI],
                fmt='^', elinewidth=4, capthick=4,
                markersize=10, markeredgewidth=3,
#                c="#A73955",
                label="GUPPI Rcvr1_2")

    ax.axvline(Time(2010.2, format='decimalyear', scale='utc').mjd, color='black', ls="--", lw=4, zorder=0)

# Plot 1
axs[0].set_ylabel("Residuals [$\mu$s]")

props = dict(boxstyle='round', facecolor='white', alpha=1.0)
axs[0].text(0.2, 0.95, "GASP", transform=axs[0].transAxes, fontsize=38,
            verticalalignment='top', bbox=props)

axs[0].text(0.7, 0.95, "GUPPI", transform=axs[0].transAxes, fontsize=38,
            verticalalignment='top', bbox=props)

# Plot 2
axs[1].legend(loc=1, bbox_to_anchor=(0.435, 1.42), framealpha=1,
              fancybox=True, fontsize=32, shadow=True, ncol=2)

axs[1].axhline(0.0, color='grey', ls="--", lw=3, zorder=0)
axs[1].set_ylabel("Averaged \n Residuals [$\mu$s] \n (Full scale)")

# Plot 3
axs[2].axhline(0.0, color='grey', ls="--", lw=3, zorder=0)
axs[2].set_ylabel("Averaged \n Residuals [$\mu$s] \n (Close-up)")
axs[2].set_xlabel("MJD [days]")
axs[2].set_ylim([-5.0, 5.0])

# Years
axtime2 = axs[0].twiny()
XLIM = axs[0].get_xlim()
XLIM = list(map(lambda x: Time(x, format='mjd', scale='utc').decimalyear, XLIM))
axtime2.set_xlim(XLIM)
axtime2.set_xlabel("Year")
axtime2.xaxis.set_major_formatter(FormatStrFormatter('%i'))

plt.tight_layout()
plt.savefig("./figures/plot_residuals.pdf")
#plt.show()
