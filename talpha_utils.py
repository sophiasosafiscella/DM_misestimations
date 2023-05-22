import astropy.units as u
import numpy as np
import pint.fitter
from astropy import log
from pint.models import get_model

from coefficients import *


def get_dmx_ranges(model, observations):
    """
    Return an array of the MJD range for each DMX parameter corresponding to a given set of observations_in_window
    """

    mjds = observations.get_mjds().value
    dmx_parameters = model.components['DispersionDMX']  # names of the DMX_xxxx parameters

    DMXR1_names = [a for a in dir(dmx_parameters) if a.startswith('DMXR1')]
    DMXR2_names = [a for a in dir(dmx_parameters) if a.startswith('DMXR2')]

    DMXR1_values = [getattr(dmx_parameters, par).value for par in DMXR1_names]
    DMXR2_values = [getattr(dmx_parameters, par).value for par in DMXR2_names]

    DMXR = np.column_stack((DMXR1_values, DMXR2_values))  # Zip the two arrays

    # Out of all the DMX windows, we will only keep those with at least one TOA inside it
    mask = (DMXR1_values < max(mjds)) & (DMXR2_values > min(mjds))
    masked_DMXR = DMXR[mask]

    return masked_DMXR


def get_dmx_observations(observations, low_mjd, high_mjd):
    """
    Return an array for selecting TOAs from toas in a DMX range.

    toas is a PINT TOA object of TOAs in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = observations.get_mjds().value
    mask = (low_mjd < mjds) & (mjds < high_mjd)
    masked_observations = observations[mask]

    return masked_observations


def broadband_observations(toas):
    """Function that, given a set of TOAs, separates the broadband observations_in_window"""

    # ----------------------------------------------------------------
    # First, we obtain the backend that was used in each epoch
    # ----------------------------------------------------------------

    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])

    # ----------------------------------------------------------------
    # Now we separate the GASP observations_in_window and the GUPPI observations_in_window
    # ----------------------------------------------------------------

    # Firstly, we mark with 'True' the GUPPI observations_in_window
    broadband_ok = np.isin(backends, ['GUPPI'])

    # Now we separate the broadband TOAs
    broadband_TOAs = toas[broadband_ok]
    n_broadband = np.count_nonzero(broadband_ok)
    print('Broadband TOAs = ' + str(n_broadband))

    return broadband_TOAs


def narrowband_observations(toas):
    """Function that, given a set of broadband observations_in_window,
     creates an artificial set of narrowband observations_in_window"""

    # ----------------------------------------------------------------
    # First, we obtain the backend and frequency that was used in each epoch
    # ----------------------------------------------------------------

    frequencies = [toas.table['freq'][obs] for obs in range(len(toas.table['freq']))]

    # We mark with 'True' the GUPPI observations_in_window with a central
    # frequency within the GASP frequency range
    narrowband_ok = np.full(len(toas), False)
    for n in range(len(frequencies)):
        if (822 <= frequencies[n] <= 866) or (1386 <= frequencies[n] <= 1434):
            narrowband_ok[n] = True

    # Now we separate the narrowband TOAs
    narrowband_TOAs = toas[narrowband_ok]
    n_narrowband = np.count_nonzero(narrowband_ok)
    print('Narrowband TOAs = ' + str(n_narrowband))

    return narrowband_TOAs


def freeze_all(model):
    for p in model.params:
        par = getattr(model, p)
        par.frozen = True

    return


def unfreeze_DMX(model):
    DMX_names = [a for a in dir(model.components['DispersionDMX']) if a.startswith('DMX_0')]

    for dmx_p in DMX_names:
        par = getattr(model, dmx_p)
        par.frozen = False

    return


def remove_some_dmx_ranges(model, toas_MJD, quiet=True):
    """
    Uses PINT to remove some DMX parameter ranges from a timing model.

    Input:
    - model: timing model
    - toas_MJD : TOAs for which you want to find DMX ranges

    model is a PINT model object.
    quiet=True turns off the logged info.
    """

    dmx_ranges, _, _ = dmx_utils.model_dmx_params(model)  # Get the range of each DMX_xxx parameter

    if 'DispersionDMX' in model.components.keys():
        dmx = model.components['DispersionDMX']
        idxs = dmx.get_indices()  # Get the indexes of the DMX_xxxx parameters
        n_removed = 0
        n_kept = 0

        for idx in idxs:  # We loop over the DMX_xxxx parameters
            remove_ok = True
            for MJD in toas_MJD:  # We loop over the GASP TOAs
                if dmx_ranges[idx - 1][0] <= MJD < dmx_ranges[idx - 1][1]:  # We check that at least one GASP TOAs
                    remove_ok = False  # is between the range of the DMX_xxxx
                    break
            if remove_ok:
                dmx.remove_DMX_range(idx)  # remove parameters
                n_removed += 1
            else:
                n_kept += 1

        if not quiet:
            msg = f"Removed {n_removed} DMX parameters from timing model."
            msg2 = f"Kept {n_kept} DMX parameters from timing model."
            log.info(msg)
            log.info(msg2)
    else:
        pass


def get_dmx_quantity(observations, model):
    """Function that fits the DMX_xxxx values for a given set of observations_in_window"""

    # Firstly, we obtain the MJD values
    observations_MJDs = observations.get_mjds().value

    # We will now remove the DispersionDMX parameters without corresponding TOAS from the list
    remove_some_dmx_ranges(model, observations_MJDs)

    # Now let's fit the DMX_xxxx parameters to this subset of TOAs
    fit = pint.fitter.GLSFitter(observations, model)
    fit.fit_toas()

    # Let us extract the fitted values for the DMX_xxxx parameters
    dmx_parameters_names = model.components['DispersionDMX'].params  # names of the DMX_xxxx parameters
    dmx_parameters = [fit.get_allparams()[parameter].quantity for parameter in dmx_parameters_names]

    return dmx_parameters


def t_alpha(parfile, broadband_observations, alpha=4.4):
    """Function that finds t_alpha = (t_2_I - t_2_J)/(q_I - q_J)
    for a given set of observations_in_window"""

    # Set the constant K with the correct units
    k_ms = 4.149 * (u.ms * (u.GHz) ** 2 * (u.cm) ** 3) / u.parsec  # ms GHz^2 pc^-1 cm^3

    dmx_quantity = []
    dmx_MJD_start = []

    # For a given DMX window, we are going to create to subset of observations_in_window: a set of broadband observations_in_window (i.e.,
    # (the full GUPPI frequency range) and a set of narroband observations_in_window (i.e., reduced to the GASP frequency range)
    observations = [narrowband_observations(broadband_observations), broadband_observations]

    for subset in observations:
        model = get_model(parfile)  # Define the model

        freeze_all(model)  # Freeze all the parameters
        unfreeze_DMX(model)  # but the DMX_xxxx parameters

        #        component, _, _, _ = model.map_component("PhaseJump") # We remove the phase jump between the two bands because
        #        model.remove_component("PhaseJump")                          # we're working with observations_in_window in only one of the bands

        dmx = get_dmx_quantity(subset,
                               model)  # Fit the DMX_xxxx parameters to only one subset of observations_in_window
        dmx_quantity.append(dmx[1::3])  # Append the fitted DMX_xxxx values.

    model = get_model(parfile)  # For the DM we are going to use the original value that came
    DM = model.components['DispersionDM'].DM.quantity  # with the model

    hat_t_2_I = k_ms * (DM - dmx_quantity[0][0])
    hat_t_2_J = k_ms * (DM - dmx_quantity[1][0])

    cGASP = lambda n: C(n, (0.822, 0.866), (1.386, 1.434))
    cGUPPI = lambda n: C(n, (0.722, 0.919), (1.151, 1.885))
    q_I = q(cGASP, alpha) * (u.GHz ** (-alpha) * u.GHz ** 2)
    q_J = q(cGUPPI, alpha) * (u.GHz ** (-alpha) * u.GHz ** 2)

    talpha = (hat_t_2_I - hat_t_2_J) / (q_I - q_J)

    return talpha
