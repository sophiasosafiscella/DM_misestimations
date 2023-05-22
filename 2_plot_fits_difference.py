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
# PSR_name = "J1643-1224"
PSR_name = "J1744-1134"
# PSR_name = "J0613-0200"
# PSR_name = "J1455-3330"
# PSR_name = "J1600-3053"
# PSR_name = "J1012+5307"
# PSR_name = "J1918-0642"
# PSR_name = "J1713+0747"

nterms = 3

sns.set_style("ticks")
sns.set_context("paper") #, font_scale=2.0, rc={"lines.linewidth": 3})
#plt.rcParams.update({"text.usetex": True})

style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 21}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 24})
rc('ytick', **{'labelsize': 24})
rc('legend', **{'fontsize': 24})
rc('axes', **{'labelsize': 22, 'titlesize': 27})
text_size = 17

fig = plt.figure(figsize=(12, 8))
#fig.suptitle(PSR_name, fontsize=20)

# Load the data_res_avg
if nterms == 2:
    gs = fig.add_gridspec(2, hspace=0)
    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_2terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_2terms.txt", skiprows=1)
elif nterms == 3:
    gs = fig.add_gridspec(3, hspace=0)
    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_3terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_3terms.txt", skiprows=1)

axs = gs.subplots(sharex=True, sharey=False)

# find the middle points
t = (narrowband_data[:, 1] + narrowband_data[:, 0]) / 2.0
t_infty_diff = broadband_data[:, 2] - narrowband_data[:, 2]
kDMX_diff = broadband_data[:, 4] - narrowband_data[:, 4]
t_alpha_diff = broadband_data[:, 6] - narrowband_data[:, 6]

# make the plots

# 1) t_infty

t_infty_diff_mean = np.mean(t_infty_diff)

axs[0].errorbar(t, t_infty_diff - t_infty_diff_mean,
                yerr=np.sqrt(broadband_data[:, 3] ** 2 + narrowband_data[:, 3] ** 2),
                c='#A083AF', fmt='o', capsize=4)
axs[0].set_ylabel("$\Delta r_\mathrm{\infty} - \overline{\Delta r_\mathrm{\infty}}$ \n $[\mathrm{\mu s}]$")
axs[0].axhline(0.0, ls='--', c='black')

at = AnchoredText(
    "$\overline{\Delta r_\mathrm{\infty}}$ = " + str(round(t_infty_diff_mean, 4)).replace("-", "$-$") + " $\mathrm{\mu s}$\n" +
    "$\sigma_{\Delta r_\mathrm{\infty}} = $" + str(round(np.std(t_infty_diff), 4)) + " $\mathrm{\mu s}$",
    prop=dict(size=text_size), frameon=True, loc='upper right')
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
                yerr=np.sqrt(broadband_data[:, 5] ** 2 + narrowband_data[:, 5] ** 2),
                c='#0FA3B1', fmt='o', capsize=4)
axs[1].set_xlabel("Window middle point [MJD]")
axs[1].set_ylabel("$\Delta r_\mathrm{2} - \overline{\Delta r_\mathrm{2}}$ \n "
                  "$[\mathrm{\mu s~GHz^2}]$")
# axs[1].set_ylim([np.amin(narrowband_data[:, 4]), np.amax(narrowband_data[:, 4])])
axs[1].axhline(0.0, ls='--', c='black')

at = AnchoredText(
    "$\overline{\Delta r_\mathrm{2}} = $" + str(round(kDMX_diff_mean, 4)).replace("-", "$-$") + " $\mathrm{\mu s}$\n" +
    "$\sigma_{\Delta r_\mathrm{2}} = $" + str(round(np.std(kDMX_diff), 4)) + " $\mathrm{\mu s}$",
    prop=dict(size=text_size), frameon=True, loc="lower center")
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
                    yerr=np.sqrt(broadband_data[:, 7] ** 2 + narrowband_data[:, 7] ** 2),
                    c='#A73955', fmt='o', capsize=4)
    axs[2].set_xlabel("Window middle point [days]")
    axs[2].set_ylabel("$\Delta r_\mathrm{\\alpha} - \overline{\Delta r_\mathrm{\\alpha}}$ \n "
                      "$[\mathrm{\mu s~GHz^{- \\alpha}}]$")
    axs[2].axhline(0.0, ls='--', c='black')

    at = AnchoredText(
        "$\overline{\Delta r_\mathrm{\\alpha}} = $" + str(round(t_alpha_diff_mean, 4)).replace("-", "$-$") + " $\mathrm{\mu s}$\n" +
        "$\sigma_{\Delta r_\mathrm{\\alpha}} = $" + str(round(np.std(t_alpha_diff), 4)) + " $\mathrm{\mu s}$",
        prop=dict(size=text_size), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[2].add_artist(at)

#    axs[2].minorticks_on()
#    axs[2].grid(which='minor', linestyle=':', linewidth='0.3', color='black')

# Hide x labels and tick labels for all but bottom plot.
for ax in axs:
    ax.label_outer()

plt.tight_layout()
plt.savefig("./figures/" + PSR_name + "_fits.pdf")
plt.show()
