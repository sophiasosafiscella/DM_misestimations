"""Pulsar timing noise models."""

import astropy.units as u
import numpy as np


# This is for the EFAC and EQUAD parameters

def sigma_scaled_cov_matrix(efac_equad, toas):
    scaled_sigma = efac_equad.scale_toa_sigma(toas).value ** 2
    return np.diag(scaled_sigma)


# This is for the ECORR parameters

def create_quantization_matrix(toas_table, dt=1, nmin=2):
    """Create quantization matrix mapping TOAs to observing epochs."""
    isort = np.argsort(toas_table)

    bucket_ref = [toas_table[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if toas_table[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas_table[i])
            bucket_ind.append([i])

    # find only epochs with more than 1 TOA
    bucket_ind2 = [ind for ind in bucket_ind if len(ind) >= nmin]

    U = np.zeros((len(toas_table), len(bucket_ind2)), "d")
    for i, l in enumerate(bucket_ind2):
        U[l, i] = 1

    return U


def ecorr_basis_weight_pair(ecorr, toas):
    """Return a quantization matrix and ECORR weights.

        A quantization matrix maps TOAs to observing epochs.
        The weights used are the square of the ECORR values.

        """
    tbl = toas.table
    t = (tbl["tdbld"].quantity * u.day).to(u.s).value
    ecorrs = ecorr.get_ecorrs()
    umats = []
    for ec in ecorrs:
        mask = ec.select_toa_mask(toas)
        if np.any(mask):
            umats.append(create_quantization_matrix(t[mask]))
        else:
#            warnings.warn(f"ECORR {ec} has no TOAs")
            umats.append(np.zeros((0, 0)))
    nc = sum(u.shape[1] for u in umats)
    umat = np.zeros((len(t), nc))
    weight = np.zeros(nc)
    nctot = 0
    for ct, ec in enumerate(ecorrs):
        mask = ec.select_toa_mask(toas)
        nn = umats[ct].shape[1]
        umat[mask, nctot: nn + nctot] = umats[ct]
        weight[nctot: nn + nctot] = ec.quantity.value ** 2
        nctot += nn
    return (umat, weight)
