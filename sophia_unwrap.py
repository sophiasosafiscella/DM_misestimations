import astropy.units as u


def unwrap(observations, model):
    # Get the original TOAs and the corresponding frequencies
    original_toas = [observations.table['mjd'][n] for n in range(len(observations.table['mjd']))]
    original_frequencies = observations.table['freq']

    # We sort them in descending order of frequency
    descending_order = original_frequencies.argsort()[::-1]
    new_toas = [original_toas[j] for j in descending_order]

    # Get the object containing the spin-down components
    spindown, _, _, _ = model.map_component("Spindown")

    for idx in range(len(new_toas)):

        while new_toas[idx] < new_toas[idx - 1]:  # If there's a jump in MJD

            # Calculate the instantaneous frequency and period
            inst_nu = spindown.F0.quantity + spindown.F1.quantity * (
                    (original_toas[idx] - spindown.PEPOCH.quantity).value * u.day).to(u.s)
            inst_period = (1.0 / inst_nu).to(u.day)

            # And add an integer number of periods
            new_toas[idx] += inst_period
            observations.table['mjd'][descending_order[idx]] += inst_period

    return observations
