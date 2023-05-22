import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import matplotlib.style as style
import seaborn as sns

import astropy.units as u
from astropy.time import Time
from astropy.visualization import quantity_support

import numpy as np
import pint
from pint.models import noise_model
import sophia_dmx

#---------------------
# Plotting paramaters
#---------------------

sns.set_style("ticks")
sns.set_context("paper") #, font_scale=2.0, rc={"lines.linewidth": 3})
style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 21}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 24})
rc('ytick', **{'labelsize': 24})
rc('axes', **{'labelsize': 27, 'titlesize': 27})

quantity_support()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# ----------------------------------------------------------------
# Global parameters
# ----------------------------------------------------------------

parfile = "./NANOGrav_12yv4/narrowband/par/J1643-1224_NANOGrav_12yv4.gls.par"
timfile = "./NANOGrav_12yv4/narrowband/tim/J1643-1224_NANOGrav_12yv4.tim"
results = open("./results/results_broadband.txt", "w+")

# ----------------------------------------------------------------
# Define the timing model, read the TOAs, and fit the model
# ----------------------------------------------------------------

model = pint.models.get_model(parfile)  # Timing model as described in the .par file
toas = pint.toa.get_TOAs(timfile, model=model)  # TOAs as described in the .tim file

# ----------------------------------------------------------------
# First, we obtain the backend that was used in each epoch
# ----------------------------------------------------------------

backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])

# ----------------------------------------------------------------
# Now we separate the GASP observations_in_window and the GUPPI observations_in_window
# ----------------------------------------------------------------

GASP_ok = np.isin(backends, ['GASP'])
GASP_toas = toas[GASP_ok]

GUPPI_ok = np.isin(backends, ['GUPPI'])
GUPPI_toas = toas[GUPPI_ok]

# ----------------------------------------------------------------
# Let's make some plots of the pre-fit original_residuals
# ----------------------------------------------------------------

ax.axhspan(0.822, 0.866, alpha=0.5, color='C3', label="GASP - Revr_800")
ax.axhspan(1.386, 1.434, alpha=0.5, color='C4', label="GASP - Revr1_2")

# Broadband observations
bb_toas = sophia_dmx.broadband_observations(toas)
rs_bb = bb_toas.table['freq'].quantity.to(u.GHz)
xt_bb = bb_toas.get_mjds()
ax.scatter(xt_bb, rs_bb, marker=".", s=50, label='GUPPI (full)',  rasterized=True)

print("Number of broadband observations: " + str(len(bb_toas)))
print("Min MJD: " + str(np.amin(xt_bb)))
print("Max MJD: " + str(np.amax(xt_bb)))
print("Min freq: " + str(np.amin(rs_bb)))
print("Max freq: " + str(np.amax(rs_bb)))
print(" ")

# Narrowband observations
nb_toas = sophia_dmx.narrowband_observations(toas)
rs_nb = nb_toas.table['freq'].quantity.to(u.GHz)
xt_nb = nb_toas.get_mjds()
ax.scatter(xt_nb, rs_nb, marker=".", s=50, label='GUPPI (narrowband)',  rasterized=True)

print("Number of narrowband observations: " + str(len(nb_toas)))
print("Min MJD: " + str(np.amin(xt_nb)))
print("Max MJD: " + str(np.amax(xt_nb)))
print("Min freq: " + str(np.amin(rs_nb)))
print("Max freq: " + str(np.amax(rs_nb)))
print(" ")

# GASP observations
# rs_GASP = Residuals(GASP_toas, model).phase_resids
rs_GASP = GASP_toas.table['freq'].quantity.to(u.GHz)
xt_GASP = GASP_toas.get_mjds()
ax.scatter(xt_GASP, rs_GASP, marker="x", s=50, label='GASP',  rasterized=True)

print("Number of GASP observations: " + str(len(GASP_toas)))
print("Min MJD: " + str(np.amin(xt_GASP)))
print("Max MJD: " + str(np.amax(xt_GASP)))
print("Min freq: " + str(np.amin(rs_GASP)))
print("Max freq: " + str(np.amax(rs_GASP)))
print(" ")

axtime2 = ax.twiny()
XLIM = ax.get_xlim()
XLIM = list(map(lambda x: Time(x, format='mjd', scale='utc').decimalyear, XLIM))
axtime2.set_xlim(XLIM)
axtime2.set_xlabel("Year")
axtime2.xaxis.set_major_formatter(FormatStrFormatter('%i'))

#ax.grid()
#ax.minorticks_on()
#ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

ax.legend(fontsize=18, loc='upper left', fancybox=True, shadow=True)
ax.set_xlabel("MJD [days]")
ax.set_ylabel("Frequency [GHz]")
#ax.grid()
plt.tight_layout()
plt.savefig("./figures/frequencies.pdf")
plt.show()
