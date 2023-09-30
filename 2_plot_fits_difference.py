# -----------------------------------------------------------------
# SECOND PROGRAM
# It creates the residual plots for the parameter fits
# -----------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import matplotlib.style as style

import numpy as np
import seaborn as sns

from astropy.time import Time


# PSR_name = "J2145-0750"
# PSR_name = "J1909-3744"
PSR_name = "J1643-1224"
# PSR_name = "J1744-1134"
# PSR_name = "J0613-0200"
# PSR_name = "J1455-3330"
# PSR_name = "J1600-3053"
# PSR_name = "J1012+5307"
# PSR_name = "J1918-0642"
# PSR_name = "J1713+0747"

nterms : int = 3
sign_figures : int = 1

sns.set_style("ticks")
sns.set_context("paper") #, font_scale=2.0, rc={"lines.linewidth": 3})
#plt.rcParams.update({"text.usetex": True})

style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 26}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 29})
rc('ytick', **{'labelsize': 29})
rc('legend', **{'fontsize': 29})
rc('axes', **{'labelsize': 27, 'titlesize': 32})
text_size = 20
leg_loc = "lower left"

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(nrows=3, ncols=4, hspace=0, wspace=0)
#fig.suptitle(PSR_name, fontsize=20)

# Load the data_res_avg
if nterms == 2:
#    gs = fig.add_gridspec(2, hspace=0)
    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_2terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_2terms.txt", skiprows=1)
elif nterms == 3:
#    gs = fig.add_gridspec(3, hspace=0)
    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_3terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_3terms.txt", skiprows=1)

#axs = gs.subplots(sharex=True, sharey=False)
axs = [fig.add_subplot(gs[0,0:3]), # large subplot (1 rows, 3 columns)
       fig.add_subplot(gs[1,0:3]), # large subplot (1 rows, 3 columns)
       fig.add_subplot(gs[2,0:3]), # large subplot (1 rows, 3 columns)
       fig.add_subplot(gs[0,3]), # small subplot (1st row, 4th column)
       fig.add_subplot(gs[1,3]), # small subplot (1st row, 4th column)
       fig.add_subplot(gs[2,3])] # small subplot (1st row, 4th column)

# find the middle points
t = (narrowband_data[:, 1] + narrowband_data[:, 0]) / 2.0
t_infty_diff = broadband_data[:, 2] - narrowband_data[:, 2]
t_infty_error = np.sqrt(broadband_data[:, 3] ** 2 + narrowband_data[:, 3] ** 2)
kDMX_diff = broadband_data[:, 4] - narrowband_data[:, 4]
kDMX_error = np.sqrt(broadband_data[:, 5] ** 2 + narrowband_data[:, 5] ** 2)
t_alpha_diff = broadband_data[:, 6] - narrowband_data[:, 6]
t_alpha_error = np.sqrt(broadband_data[:, 7] ** 2 + narrowband_data[:, 7] ** 2)
# make the plots

# 1) t_infty

t_infty_diff_mean = np.mean(t_infty_diff)

axs[0].errorbar(t, t_infty_diff - t_infty_diff_mean,
                yerr=t_infty_error,
                c='#D60270', fmt='o', capsize=4)
axs[0].set_ylabel("$\Delta r_\mathrm{\infty} - \overline{\Delta r_\mathrm{\infty}}$ \n $[\mathrm{\mu s}]$")
axs[0].axhline(0.0, ls='--', c='black')
axs[0].set_ylim([-50.0, 50.0])

at = AnchoredText(
    "$\overline{\Delta r_\mathrm{\infty}}$ = " + str(round(t_infty_diff_mean, sign_figures)).replace("-", "$-$") + " $\mathrm{\mu s}$\n" +
    "$\sigma_{\Delta r_\mathrm{\infty}} = $" + str(round(np.std(t_infty_diff), sign_figures)) + " $\mathrm{\mu s}$",
    prop=dict(size=text_size), frameon=True, loc=leg_loc)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
axs[0].add_artist(at)

axtime2 = axs[0].twiny()
XLIM = axs[0].get_xlim()
XLIM = list(map(lambda x: Time(x, format='mjd', scale='utc').decimalyear, XLIM))
axtime2.set_xlim(XLIM)
axtime2.set_xlabel(r"$\mathrm{Year}$")
axtime2.xaxis.set_major_formatter(FormatStrFormatter('%i'))

# 2) kDMX

kDMX_diff_mean = np.mean(kDMX_diff)

# if PSR_name == "J1455-3330" or PSR_name == "J1918-0642":
if PSR_name == "J1455-3330":
    delta = kDMX_diff - kDMX_diff_mean
    ind = np.where(abs(delta) > 70.0)

    broadband_data = np.delete(broadband_data, ind, 0)
    narrowband_data = np.delete(narrowband_data, ind, 0)
    np.savetxt("./" + PSR_name + "_narrowband/fit_dispersion_results_3terms.txt", narrowband_data)
    np.savetxt("./" + PSR_name + "_broadband/fit_dispersion_results_3terms.txt", broadband_data)

    t = (narrowband_data[:, 1] + narrowband_data[:, 0]) / 2.0
    t_infty_diff = broadband_data[:, 2] - narrowband_data[:, 2]
    kDMX_diff = broadband_data[:, 4] - narrowband_data[:, 4]
    t_alpha_diff = broadband_data[:, 6] - narrowband_data[:, 6]

axs[1].errorbar(t, kDMX_diff - kDMX_diff_mean,
                yerr=kDMX_error,
                c='#9B4F96', fmt='o', capsize=4)
axs[1].set_xlabel("Window middle point [MJD]")
axs[1].set_ylabel("$\Delta r_\mathrm{2} - \overline{\Delta r_\mathrm{2}}$ \n "
                  "$[\mathrm{\mu s~GHz^2}]$")
# axs[1].set_ylim([np.amin(narrowband_data[:, 4]), np.amax(narrowband_data[:, 4])])
axs[1].axhline(0.0, ls='--', c='black')
axs[1].set_yticks([-50.0, 50.0])

at = AnchoredText(
    "$\overline{\Delta r_\mathrm{2}} = $" + str(round(kDMX_diff_mean, sign_figures)).replace("-", "$-$") + " $\mathrm{\mu s}$\n" +
    "$\sigma_{\Delta r_\mathrm{2}} = $" + str(round(np.std(kDMX_diff), sign_figures)) + " $\mathrm{\mu s}$",
    prop=dict(size=text_size), frameon=True, loc="upper left")
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
axs[1].add_artist(at)

#axs[0].minorticks_on()
#axs[0].grid(which='minor', linestyle=':', linewidth='0.3', color='black')

#axs[1].minorticks_on()
#axs[1].grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# 3) t_alpha

if nterms == 3:
    t_alpha_diff_mean = np.mean(t_alpha_diff)
    axs[2].errorbar(t, t_alpha_diff - t_alpha_diff_mean,
                    yerr=t_alpha_error,
                    c='#0038A8', fmt='o', capsize=4)
    axs[2].set_xlabel("Window middle point [days]")
    axs[2].set_ylabel("$\Delta r_\mathrm{\\alpha} - \overline{\Delta r_\mathrm{\\alpha}}$ \n "
                      "$[\mathrm{\mu s~GHz^{- \\alpha}}]$")
    axs[2].axhline(0.0, ls='--', c='black')

    at = AnchoredText(
        "$\overline{\Delta r_\mathrm{\\alpha}} = $" + str(round(t_alpha_diff_mean, sign_figures)).replace("-", "$-$") + " $\mathrm{\mu s}$\n" +
        "$\sigma_{\Delta r_\mathrm{\\alpha}} = $" + str(round(np.std(t_alpha_diff), sign_figures)) + " $\mathrm{\mu s}$",
        prop=dict(size=text_size), frameon=True, loc=leg_loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[2].add_artist(at)
    axs[2].set_xticks([55500, 56500, 57500])
    axs[2].set_xticks([56000, 57000], minor=True)

#    axs[2].minorticks_on()
#    axs[2].grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# Histograms

axs[3].hist(np.divide(t_infty_diff - t_infty_diff_mean, t_infty_error), orientation='horizontal')
axs[3].yaxis.tick_right()
axs[3].yaxis.set_label_position("right")
axs[3].set_ylim([-3.5, 3.5])
axs[3].set_yticks([-2.0, 0.0, 2.0])
axs[3].set_yticks([-3.0, -1.0, 0.0, 1.0, 3.0], minor=True)
axs[3].label_outer()

axs[4].hist(np.divide(kDMX_diff - kDMX_diff_mean, kDMX_error), bins=6, orientation='horizontal')
axs[4].yaxis.tick_right()
axs[4].set_ylabel("Residual/Uncertainty", labelpad=10)
axs[4].yaxis.set_label_position("right")
axs[4].set_ylim([-3.5, 3.5])
axs[4].set_yticks([-2.0, 0.0, 2.0])
axs[4].set_yticks([-3.0, -1.0, 0.0, 1.0, 3.0], minor=True)
axs[4].label_outer()

axs[5].hist(np.divide(t_alpha_diff - t_alpha_diff_mean, t_alpha_error),
#            bins=36,
            orientation='horizontal')
axs[5].yaxis.tick_right()
axs[5].yaxis.set_label_position("right")
axs[5].set_ylim([-3.5, 3.5])
axs[5].set_xticks([10, 30])
axs[5].set_xticks([20, 40], minor=True)
axs[5].set_yticks([-2.0, 0.0, 2.0])
axs[5].set_yticks([-3.0, -1.0, 0.0, 1.0, 3.0], minor=True)
axs[5].set_xlabel("Number of \n measurements")

# Hide x labels and tick labels for all but bottom plot.
for ax in axs[0:2]:
    ax.label_outer()

plt.tight_layout()
plt.savefig("./figures/" + PSR_name + "_fits.pdf")
plt.show()
