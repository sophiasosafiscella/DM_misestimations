import sys
import time

import astropy.units as u
import numpy as np
import pint.fitter
from astropy import log
from pint.models import get_model
from pint.toa import get_TOAs
from scipy import linalg

from coefficients import *
from sophia_cov_matrix import sigma_scaled_cov_matrix, ecorr_basis_weight_pair


def remove_some_dmx_ranges(model, toas_MJD, quiet=False):
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


def old_get_dmx_quantity(toas, model, backend):
    """Function that fits the DMX_xxxx values for a given backend"""

    t0 = time.time()

    # First, we obtain the backend that was used in each epoch
    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])

    # Firstly, we mark with 'True' the observations_in_window with the desired backend
    backend_ok = np.isin(backends, [backend])
    backend_TOAs = toas[backend_ok]
    backend_TOAs_MJDs = backend_TOAs.get_mjds().value

    # We will now remove the DispersionDMX parameters without corresponding TOAS from the list
    remove_some_dmx_ranges(model, backend_TOAs_MJDs)

    # Now let's fit the DMX_xxxx parameters to this subset of TOAs
    fit = pint.fitter.GLSFitter(backend_TOAs, model)
    fit.fit_toas()

    # Let us extract the fitted values for the DMX_xxxx parameters
    dmx_parameters_names = model.components['DispersionDMX'].params  # names of the DMX_xxxx parameters
    dmx_parameters = [fit.get_allparams()[parameter].quantity for parameter in dmx_parameters_names]

    t1 = time.time()
    print("get_dmx_quantities: " + str(t1 - t0))
    return dmx_parameters


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


def old_t_alpha(timfile, parfile, alpha=4.4):
    """Function that finds t_alpha = (t_2_I - t_2_J)/(q_I - q_J)"""

    # Set the constant K with the correct units
    k_ms = (4.149 * u.ms)  # ms GHz^2 pc^-1 cm^3
    k_days = k_ms.to(u.day)  # days GHz^2 pc^-1 cm^3
    k = k_days * ((u.GHz) ** 2 * (u.cm) ** 3) / u.parsec

    dmx_quantity = []
    dmx_MJD_start = []

    for backend in ['GASP', 'GUPPI']:
        model = get_model(parfile)  # Define the model
        toas = pint.toa.get_TOAs(timfile, model=model)  # Get the TOAs
        dmx = old_get_dmx_quantity(toas, model, backend)  # Fit the DMX_xxxx parameters for only one of the backends
        dmx_quantity.append(dmx[1::3])  # Append the fitted DMX_xxxx values.
        dmx_MJD_start.append(dmx[2::3])

    model = get_model(parfile)
    DM = model.components['DispersionDM'].DM.quantity

    for idx1 in range(len(dmx_MJD_start[0])):
        for idx2 in range(len(dmx_MJD_start[1])):
            if dmx_MJD_start[0][idx1] == dmx_MJD_start[1][idx2]:  # Find a matching time window between backends
                hat_t_2_I = k * (DM - dmx_quantity[0][idx1])
                hat_t_2_J = k * (DM - dmx_quantity[1][idx2])
                break
        else:
            continue  # When we find a match, break the loop
        break

    cGASP = lambda n: C(n, (0.822, 0.866), (1.386, 1.434))
    cGUPPI = lambda n: C(n, (0.722, 0.919), (1.151, 1.885))
    q_I = q(cGASP, alpha) * (u.GHz ** (-alpha) * u.GHz ** 2)
    q_J = q(cGUPPI, alpha) * (u.GHz ** (-alpha) * u.GHz ** 2)

    talpha = (hat_t_2_I - hat_t_2_J) / (q_I - q_J)
    return talpha


def t_alpha(parfile, observations, alpha=4.4):
    """Function that finds t_alpha = (t_2_I - t_2_J)/(q_I - q_J)
    for a given set of observations_in_window"""

    # Set the constant K with the correct units
    k_ms = (4.149 * u.ms)  # ms GHz^2 pc^-1 cm^3
    k_days = k_ms.to(u.day)  # days GHz^2 pc^-1 cm^3
    k = k_days * ((u.GHz) ** 2 * (u.cm) ** 3) / u.parsec

    dmx_quantity = []
    dmx_MJD_start = []

    for subset in observations:
        model = get_model(parfile)  # Define the model
        dmx = get_dmx_quantity(subset, model)  # Fit the DMX_xxxx parameters for only one of the backends
        dmx_quantity.append(dmx[1::3])  # Append the fitted DMX_xxxx values.
        dmx_MJD_start.append(dmx[2::3])

    model = get_model(parfile)
    DM = model.components['DispersionDM'].DM.quantity

    for idx1 in range(len(dmx_MJD_start[0])):
        for idx2 in range(len(dmx_MJD_start[1])):
            if dmx_MJD_start[0][idx1] == dmx_MJD_start[1][idx2]:  # Find a matching time window between backends
                hat_t_2_I = k * (DM - dmx_quantity[0][idx1])
                hat_t_2_J = k * (DM - dmx_quantity[1][idx2])
                break
        else:
            continue  # When we find a match, break the loop
        break

    cGASP = lambda n: C(n, (0.822, 0.866), (1.386, 1.434))
    cGUPPI = lambda n: C(n, (0.722, 0.919), (1.151, 1.885))
    q_I = q(cGASP, alpha) * (u.GHz ** (-alpha) * u.GHz ** 2)
    q_J = q(cGUPPI, alpha) * (u.GHz ** (-alpha) * u.GHz ** 2)

    talpha = (hat_t_2_I - hat_t_2_J) / (q_I - q_J)
    return talpha


def t_PE(nu: float, model):
    """ Profile evolution. Frequencies must be in GHz """

    FD = [model.get_params_dict()['FD1'].quantity.to(u.day), model.get_params_dict()['FD2'].quantity.to(u.day)]

    return FD[0] * math.log10(nu.value) + FD[1] * (math.log10(nu.value)) ** 2


def t_C(nu, talpha: float, alpha=4.4):
    """Chromatic term"""

    return talpha * nu ** (-1.0 * alpha)

def find_DMX(model, fit, observation_MJD):
    """Finds the DMX_xxxx value with the correct window for a given MJD"""

    # Let us extract the fitted values for the DMX_xxxx parameters
    dmx_parameters = model.components['DispersionDMX'].params[1::3]  # names of the DMX_xxxx parameters
    dmx_parameters_quantities = [fit.get_allparams()[parameter].quantity for parameter in dmx_parameters]
    dmx_ranges, _, _ = dmx_utils.model_dmx_params(model)

    dmx = 0  # Initialize the value as zero
    found = False  # We will keep track of a match

    for idx in range(len(dmx_parameters_quantities)):
        if dmx_ranges[idx][0] <= observation_MJD < dmx_ranges[idx][1]:
            dmx = dmx_parameters_quantities[idx]
            found = True
            break
        else:
            continue

    if not found:
        print("ERROR: Could not find a proper DMX_xxxx value for MJD = " + str(observation_MJD))
        sys.exit()

    return dmx


def narrowband_observations(toas):
    """Function that, given a set of broadband observations_in_window,
     creates an artificial set of narrowband observations_in_window"""

    # ----------------------------------------------------------------
    # First, we obtain the backend and frequency that was used in each epoch
    # ----------------------------------------------------------------

    frequencies = [toas.table['freq'][obs] for obs in range(len(toas.table['freq']))]
    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])

    # We mark with 'True' the GUPPI observations_in_window with a central
    # frequency within the GASP frequency range
    narrowband_ok = np.full(len(toas), False)
    for n in range(len(frequencies)):
        if ((822 <= frequencies[n] <= 866) or (1386 <= frequencies[n] <= 1434)) and backends[n] == 'GUPPI':
            narrowband_ok[n] = True

    # Now we separate the narrowband TOAs
    narrowband_TOAs = toas[narrowband_ok]
    n_narrowband = np.count_nonzero(narrowband_ok)
    print('Narrowband TOAs = ' + str(n_narrowband))

    return narrowband_TOAs


def old_narrowband_observations(toas):
    """Function that, given a set of TOAs, separates the narrowband observations_in_window"""

    # ----------------------------------------------------------------
    # First, we obtain the backend and frequency that was used in each epoch
    # ----------------------------------------------------------------

    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])
    frequencies = [toas.table['freq'][obs] for obs in range(len(toas.table['freq']))]

    # ----------------------------------------------------------------
    # Now we separate the GASP observations_in_window and the GUPPI observations_in_window
    # ----------------------------------------------------------------

    # Firstly, we mark with 'True' the GASP observations_in_window
    GASP_ok = np.isin(backends, ['GASP'])

    # Secondly, we mark with 'True' the GUPPI observations_in_window with a central
    # frequency within the GASP frequency range
    GUPPI_narrowband_ok = np.full(len(GASP_ok), False)
    GUPPI_ok = np.isin(backends, ['GUPPI'])
    for n in range(len(frequencies)):
        if ((822 <= frequencies[n] <= 866) or (1386 <= frequencies[n] <= 1434)) and GUPPI_ok[n]:
            GUPPI_narrowband_ok[n] = True

    # Lastly, merge the two sets of observations_in_window
    narrowband_ok = np.logical_or(GASP_ok, GUPPI_narrowband_ok)

    # Now we separate the narrowband TOAs
    narrowband_TOAs = toas[narrowband_ok]
    n_narrowband = np.count_nonzero(narrowband_ok)
    print('Narrowband TOAs = ' + str(n_narrowband))

    return narrowband_TOAs


def broadband_observations(toas):
    """Function that, given a set of TOAs, separates the narrowband observations_in_window"""

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
    print('Narrowband TOAs = ' + str(n_broadband))

    return broadband_TOAs


def t_infty(observations, talpha, alpha, model, fit):
    # Set the constant K with the correct units
    k_ms = (4.149 * u.ms)  # ms GHz^2 pc^-1 cm^3
    k_days = k_ms.to(u.day)  # days GHz^2 pc^-1 cm^3
    k = k_days * ((u.GHz) ** 2 * (u.cm) ** 3) / u.parsec

    # Set the frequencies to GHz
    frequencies_MHz = observations.table['freq'].quantity
    frequencies_GHz = frequencies_MHz.to(u.GHz)  # Convert frequencies to GHz

    # Set the TOAs to days
    mjds_day = [toa.value * u.day for toa in observations.get_mjds(high_precision=True)]

    tinfty = []

    DM = model.components['DispersionDM'].DM.quantity  # Units:  pc / cm3

    for t, nu in zip(mjds_day, frequencies_GHz):
        dmx: float = find_DMX(model, fit, t.value)  # Units: pc / cm3
        tinfty2 = t - (k * (DM - dmx)) / (nu ** 2) - t_PE(nu, model) - t_C(nu, talpha, alpha)
        tinfty.append(tinfty2.value)

    return tinfty


def t_infty_hat(model, observations, tinfty):
    # Get the EFAC, EQUAD, and ECORR parameters from the .par file
    efac_equad, _, _, _ = model.map_component("ScaleToaError")
    ecorr, _, _, _ = model.map_component("EcorrNoise")

    # We calculate the covariance matrix for EFAC and EQUAD, and then for ECORR, alongside the exploder matrix U
    efac_equad_matrix = sigma_scaled_cov_matrix(efac_equad, observations)
    U, ecorr_matrix_1x1 = ecorr_basis_weight_pair(ecorr, observations)

    # By adding these two together, we get the covariance matrix
    C = efac_equad_matrix + ecorr_matrix_1x1[0]

    # We calculate the estimate for t_infty as the weighted average of the t_infty estimates for the different epochs
    A = np.dot(np.dot(linalg.inv(np.dot(np.dot(U.T, linalg.inv(C)), U)), U.T), linalg.inv(C))
    tinftyhat = np.dot(A, [t for t in tinfty])

    return tinftyhat[0]
