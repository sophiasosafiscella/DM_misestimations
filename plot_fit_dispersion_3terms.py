import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.style as style
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

import numpy as np
import astropy.units as u
import string

from lmfit import Model, Parameters
from pint.models import get_model
from pint.residuals import Residuals
from pint.toa import get_TOAs

import sophia_dmx
import talpha_utils

# Dispersion curve we will be fitting to the residuals
def dispersion_curve_3terms(frequencies, t_infty, k_DMX, t_alpha):
    alpha = 4.4
    t_nu = np.array([t_infty + k_DMX / (nu ** 2) + t_alpha / (nu ** alpha) for nu in frequencies])

    return t_nu


# ----------------------------------------------------------------
# Global parameters
# ----------------------------------------------------------------

sns.set_style("ticks")
sns.set_context("paper")
style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 21}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 24})
rc('ytick', **{'labelsize': 24})
rc('legend', **{'fontsize': 20})
rc('axes', **{'labelsize': 27, 'titlesize': 27})

PSR_name = "J1643-1224"

fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharex=True, sharey=True,
                        gridspec_kw={'wspace': 0, 'hspace': 0})
#plt.suptitle(PSR_name)

# Input files
parfile = "./NANOGrav_12yv4/narrowband/par/" + PSR_name + "_NANOGrav_12yv4.gls.par"
timfile = "./NANOGrav_12yv4/narrowband/tim/" + PSR_name + "_NANOGrav_12yv4.tim"


for i, (dataset, ax) in enumerate(zip(["Broadband", "Narrowband"], axs)):

    # ----------------------------------------------------------------
    # Define the timing model, read the TOAs, and fit the model
    # ----------------------------------------------------------------

    timing_model = get_model(parfile)  # Timing model as described in the .par file
    toas = get_TOAs(timfile)  # TOAs as described in the .tim file

    # Now we will separate the narrowband observations
    if dataset == "Narrowband":
        new_toas = sophia_dmx.narrowband_observations(toas)  # Keep only the narrowband observations
    elif dataset == "Broadband":
        new_toas = sophia_dmx.broadband_observations(toas)

    # Find the DMX windows
    dmx_ranges = talpha_utils.get_dmx_ranges(timing_model, new_toas)

    # Get rid of the DMX parameters
    timing_model.remove_component("DispersionDMX")

    # For each window
    for n, window in enumerate([dmx_ranges[1]]):
        print("Working on " + str(window[0]) + " to " + str(window[1]) + "...")

        # We find the observations in the windows and the corresponding frequencies
        observations_in_window = talpha_utils.get_dmx_observations(new_toas, window[0], window[1])
        frequencies = np.array([observations_in_window.table['freq'][obs]
                                for obs in range(len(observations_in_window.table['freq']))])

        # We make sure there are observations in both bands
        lowerband_ok = np.any((822 <= frequencies) & (frequencies <= 866))
        upperband_ok = np.any((1386 <= frequencies) & (frequencies <= 1434))

        if (not lowerband_ok) or (not upperband_ok):
            print("Skipped an window because it didn't have observations in both bands")
            continue

        # ----------------------------------------------------------------
        # Calculate residuals (in microseconds) with the simplified model
        # ----------------------------------------------------------------

        original_residuals = Residuals(observations_in_window, timing_model).time_resids.to(u.us).value
        #    xt = [toa.value for toa in observations_in_window.get_mjds(high_precision=True)]
        frequencies_MHz = observations_in_window.table['freq'].quantity
        frequencies_GHz = frequencies_MHz.to(u.GHz).value  # Convert frequencies to GHz

        # ----------------------------------------------------------------
        # Fit these residuals to our dispersion curve
        # ----------------------------------------------------------------

        dispersion_model = Model(dispersion_curve_3terms)

        # create a set of Parameters
        params = Parameters()
        params.add('t_infty', value=0.0, vary=True)
        params.add('k_DMX', value=0.0, vary=True)
        params.add('t_alpha', value=2.0, min=0.0, max=10.0, vary=True)

        # do fit, here with the default leastsq algorithm
        result = dispersion_model.fit(original_residuals, params=params, frequencies=frequencies_GHz)

        frequencies_to_plot = np.linspace(0.7, 1.9)
        fitted_function = dispersion_curve_3terms(frequencies_to_plot, result.best_values['t_infty'],
                                                  result.best_values['k_DMX'],
                                                  result.best_values['t_alpha'])

        # Include the fit results in a text box
        at = AnchoredText(
            "$r_\mathrm{\infty}$ = " + str(round(result.params['t_infty'].value, 2)).replace("-", "$-$") + "$\pm$"
            + str(round(result.params['t_infty'].stderr, 2)) + " $\mu$s \n" +
            "$r_\mathrm{2}$ = " + str(round(result.best_values['k_DMX'], 2)) + "$\pm$"
            + str(round(result.params['k_DMX'].stderr, 2)) + " $\mu$s $\mathrm{GHz^2}$ \n" +
            "$r_\mathrm{\\alpha}$ = " + str(round(result.best_values['t_alpha'], 2)) + "$\pm$"
            + str(round(result.params['t_alpha'].stderr, 2)) + " $\mu$s $\mathrm{GHz^{- \\alpha}}$",
            prop=dict(size=20), frameon=True, loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        # Label the subplots
        ax.text(0.07, 0.92, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes)

        # Plot
        ax.plot(frequencies_to_plot, fitted_function, c='C1',
                label='$r_\mathrm{t_\infty} + r_\mathrm{2} \\nu^{-2} + r_\mathrm{\\alpha} \\nu^{\\alpha}$')
        ax.plot(frequencies_GHz, original_residuals, "o", markersize=8,c='C0')


#        ax.grid()
        ax.set_title(dataset)
        ax.set_xlabel("Frequency [GHz]")
        ax.tick_params(axis='both', which='major')

        if dataset == "Broadband":
            ax.set_ylabel('Residuals [$\mu s$]')


        ax.axvspan(0.822, 0.866, alpha=0.4, color='C2', label="GASP - Revr_800")
        ax.axvspan(1.386, 1.434, alpha=0.4, color='C3', label="GASP - Revr1_2")

        if i==1:
            ax.legend(loc=1, bbox_to_anchor=(1.0, 0.71), fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('./figures/fits.pdf')
plt.show()

print("Done!")
