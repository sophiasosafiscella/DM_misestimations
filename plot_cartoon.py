
import numpy as np
from matplotlib.pyplot import *
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, MultipleLocator
import scipy.optimize as optimize
import sys
#sys.path.append("/home/michael/Research/Noise Budget/source/")
#import utilities as u
from uncertainties import ufloat



rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Times New Roman'],'size':14})#,'weight':'bold'})
rc('xtick',**{'labelsize':16})
rc('ytick',**{'labelsize':16})
rc('axes',**{'labelsize':18,'titlesize':18})


def nolog(x,pos):
    y = np.log10(x)
    if y == 1:
        return "$\hfill 10$"
    elif y == 0:
        return "$\hfill 1$"
    elif y == -1:
        return "$\hfill 0.1$"
    elif y == -2:
        return "$\hfill 0.01$"
    return "$\hfill 10^{%i}$" % np.log10(x)
formatter = FuncFormatter(nolog)

SEED = 42


def make_observation(nus,tDM,tS,nu0=1.0):
    nus2 = (nus/nu0)**-2
    nus4p4 = (nus/nu0)**-4.4

    toas = tDM*nus2 + tS*nus4p4
    return toas



def fitfunc(p,nu):
    return p[0] + p[1]/nu**2
def errfunc(p,nu,y):
    return y - fitfunc(p,nu)
pinit = [1.0,1.0]

def fit_DM(nus,toas): #fit DM on a single epoch
    out = optimize.leastsq(errfunc,pinit,args=(nus,toas),full_output=1)
    resids = toas - fitfunc(out[0],nus)
    s_sq = (errfunc(out[0],nus,toas)**2).sum()/(len(toas)-len(pinit))

    tinf = ufloat(out[0][0],np.sqrt(out[1][0,0]*s_sq))
    tDMhat = ufloat(out[0][1],np.sqrt(out[1][1,1]*s_sq))

    return resids,tinf,tDMhat


def get_excess_noise(resids): #resids can be 1d or 2d
    N = np.size(resids)
    return np.sum(resids**2)/N



tDM = 1.0
tS = 0.1


tDM = 1.0
tS = 0.01

#tDM = 0.1
#tS = 1.0

dnu = 0.004 #this does affect the error bars

def arange(start,stop,delta):
    return np.arange(start,stop+delta,delta)

nus_full = arange(0.7,1.9,dnu)
toas_full = make_observation(nus_full,tDM,tS)
resids_full,tinf_full,tDMhat_full = fit_DM(nus_full,toas_full)

nus_guppi = np.concatenate((arange(0.722,0.919,dnu),arange(1.151,1.885,dnu)))
toas_guppi = make_observation(nus_guppi,tDM,tS)
resids_guppi,tinf_guppi,tDMhat_guppi = fit_DM(nus_guppi,toas_guppi)

nus_gasp = np.concatenate((arange(0.822,0.866,dnu),arange(1.386,1.434,dnu)))
toas_gasp = make_observation(nus_gasp,tDM,tS)
resids_gasp,tinf_gasp,tDMhat_gasp = fit_DM(nus_gasp,toas_gasp)

print("t_inf,tDMhat,excess noise")
print(tinf_full,tDMhat_full,get_excess_noise(resids_full))
print(tinf_guppi,tDMhat_guppi,get_excess_noise(resids_guppi))
print(tinf_gasp,tDMhat_gasp,get_excess_noise(resids_gasp))


fig = figure(figsize=(6, 6))
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True)

ax1, ax2 = axs[0], axs[1]


ax1.plot(nus_full, toas_full, '-', c='k', label=r'$K \times \mathrm{DM} = %0.2f~\mathrm{GHz}, t_{C} = %0.2f~\mathrm{GHz}$'%(tDM,tS))
ax1.plot(nus_full, make_observation(nus_full,tDM,0.0,nu0=1.0), '--', c='0.50', label=r'$K \times \mathrm{DM} = %0.2f~\mathrm{GHz}$'%(tDM,))
ax1.set_ylabel(r'$t_{\nu,\mathrm{obs}}~(\mu \mathrm{s})$')
ax1.legend()

ax2.plot(nus_full, resids_full, '.', ms=4, label='Full Band')
ax2.plot(nus_guppi, resids_guppi, '+', label='GUPPI')
ax2.plot(nus_gasp, resids_gasp, 'x', label='GASP')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel(r'$\Delta t~(\mu \mathrm{s})$')
ax2.tick_params(axis='both', which='major')
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax2.legend()

#ax2.set_title(r'$\Delta t_{\rm DM,1~GHz} = %0.2f~\mathrm{\mu s}, \Delta t_{\rm S,1~GHz} = %0.2f~\mathrm{\mu s}$'%(tDM,tS))
#legend()
fig.tight_layout()
savefig("cartoon2.png")
savefig("cartoon2.pdf")
show()
