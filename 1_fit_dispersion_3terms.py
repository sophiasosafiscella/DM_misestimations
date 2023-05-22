# -----------------------------------------------------------------
# FIRST AND MAIN PROGRAM
# It reads the TOAs, creates the broadband and narrowband datasets,
# and performs the fits using three parameters
# -----------------------------------------------------------------

import os
import sys

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.style as style
from matplotlib.offsetbox import AnchoredText

import numpy as np
import seaborn as sns
from astropy import units as u


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

plot = False

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

# Input files
parfile = "./NANOGrav_12yv4/narrowband/par/" + PSR_name + "_NANOGrav_12yv4.gls.par"
timfile = "./NANOGrav_12yv4/narrowband/tim/" + PSR_name + "_NANOGrav_12yv4.tim"

print("parfile : " + parfile)
print("timefile : " + timfile)

sns.set_theme(style='darkgrid')

for dataset in ["broadband", "narrowband"]:

    n_observations: int = 0

    # Check if the folder to save the results exists. If not, create it.
    path = "./results/" + PSR_name + "_" + dataset
    if not os.path.exists(path):
        os.makedirs(path)

    # Define the timing model, read the TOAs, and fit the model
    timing_model = get_model(parfile)  # Timing model as described in the .par file
    toas = get_TOAs(timfile)  # TOAs as described in the .tim file

    # Output file
    out_file = open(path + "/fit_dispersion_results_3terms.txt", "w+")
    out_file.writelines("DMXR1    DMXR2    t_infty [ms]     k*DMX [ms GHz^2]     t_alpha [ms GHz^-alpha]\n")

    # Now we will separate the narrowband observations
    if dataset == "narrowband":
        new_toas = sophia_dmx.narrowband_observations(toas)  # Keep only the narrowband observations
    elif dataset == "broadband":
        new_toas = sophia_dmx.broadband_observations(toas)

    # Find the DMX windows
    dmx_ranges = talpha_utils.get_dmx_ranges(timing_model, new_toas)
    range_lengths = [(range[1] - range[0]) for range in dmx_ranges]

    #  Fix the nominal DM values by adding DMX_0001 to DM (only for some pulsars)
    if PSR_name == "J1918-0642" or PSR_name == "J1012+5307":
        timing_model.components['DispersionDM'].DM.quantity += timing_model.components[
            'DispersionDMX'].DMX_0001.quantity

    # Get rid of the DMX parameters
    timing_model.remove_component("DispersionDMX")

    # For each window
    for n, window in enumerate(dmx_ranges):
        print("Working on " + str(window[0]) + " to " + str(window[1]) + " (index " + str(n) + ")...")

        # We find the observations in the windows and the corresponding frequencies
        observations_in_window = talpha_utils.get_dmx_observations(new_toas, window[0], window[1])
        frequencies = np.array([observations_in_window.table['freq'][obs]
                                for obs in range(len(observations_in_window.table['freq']))])

        # We make sure there are observations in both bands
        lowerband_ok = np.any((822 <= frequencies) & (frequencies <= 866))
        upperband_ok = np.any((1386 <= frequencies) & (frequencies <= 1434))

        if (not lowerband_ok) or (not upperband_ok):
            print("Skipped a window because it didn't have observations in both bands")
            continue
        else:
            n_observations += len(observations_in_window)

        # ----------------------------------------------------------------
        # Calculate residuals (in microseconds) with the simplified model
        # ----------------------------------------------------------------

        original_residuals = Residuals(observations_in_window, timing_model).time_resids.to(u.us).value
        frequencies_MHz = observations_in_window.table['freq'].quantity
        frequencies_GHz = frequencies_MHz.to(u.GHz).value  # Convert frequencies to GHz

        # ----------------------------------------------------------------
        # Fit these residuals to our dispersion curve
        # ----------------------------------------------------------------

        dispersion_model = Model(dispersion_curve_3terms)

        # create a set of Parameters
        params = Parameters()
        params.add('t_infty', value=20.0, vary=True)
        params.add('k_DMX', value=-20.0, vary=True)
        params.add('t_alpha', value=2.0, min=0.0, max=10.0, vary=True)

        # do fit, here with the default leastsq algorithm
        result = dispersion_model.fit(original_residuals, params=params, frequencies=frequencies_GHz)

        # write the results to the output file
        out_file.writelines(str(window[0]) + ' ' + str(window[1]) + ' '
                            + str(result.params['t_infty'].value) + ' '
                            #                        + str(result.best_values['t_infty']) + ' '
                            + str(result.params['t_infty'].stderr) + ' '
                            + str(result.params['k_DMX'].value) + ' '
                            #                        + str(result.best_values['k_DMX']) + ' '
                            + str(result.params['k_DMX'].stderr) + ' '
                            + str(result.params['t_alpha'].value) + ' '
                            #                        + str(result.best_values['k_DMX']) + ' '
                            + str(result.params['t_alpha'].stderr) + '\n')

        frequencies_to_plot = np.linspace(np.amin(frequencies_GHz), np.amax(frequencies_GHz))
        fitted_function = dispersion_curve_3terms(frequencies_to_plot, result.best_values['t_infty'],
                                                  result.best_values['k_DMX'],
                                                  result.best_values['t_alpha'])

        # Make some cool plots
        fig, ax = plt.subplots()

        at = AnchoredText(
            "$r_\mathrm{\infty}$ = " + str(round(result.params['t_infty'].value, 4)) + "$\pm$"
            + str(round(result.params['t_infty'].stderr, 4)) + " $\mu$s \n" +
            "$r_\mathrm{2}$ = " + str(round(result.params['k_DMX'].value, 4)) + "$\pm$"
            + str(round(result.params['k_DMX'].stderr, 4)) + " $\mu$s $\mathrm{GHz^2}$ \n" +
            "$r_\mathrm{\\alpha}$ = " + str(round(result.params['t_alpha'].value, 4)) + "$\pm$"
            + str(round(result.params['t_alpha'].stderr, 4)) + " $\mu$s $\mathrm{GHz^{- \\alpha}}$",
            prop=dict(size=10), frameon=True, loc='lower left')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        ax.plot(frequencies_GHz, original_residuals, "o")
        ax.plot(frequencies_to_plot, fitted_function,
                label='$r(\\nu) = r_\mathrm{\infty} + r_\mathrm{2} \\nu^{-2} + r_\mathrm{\\alpha} \\nu^{\\alpha}$')

        ax.set_title("MJD " + str(window[0]) + " to " + str(window[1]))
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel(r'Residuals [$\mu s$]')

        ax.axvspan(0.822, 0.866, alpha=0.4, color='C2', label="GASP - Revr_800")
        ax.axvspan(1.386, 1.434, alpha=0.4, color='C3', label="GASP - Revr1_2")

        ax.legend(loc="upper right")
        plt.tight_layout()

        if plot:
            plt.savefig(path + '/fit_' + str(n) + '.png')

        plt.show()

        print("Done!")

    print("Number of observations = " + str(n_observations))

    out_file.close()
