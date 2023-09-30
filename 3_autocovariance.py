import string

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.style as style

import numpy as np
import seaborn as sns
from lmfit import Model, Parameters
from matplotlib.offsetbox import AnchoredText


def acf_new(times, x, month=60.0):
    length = len(x)
    num_lags = int(np.ceil((np.max(times) - np.min(times)) / month) + 1)  # +1?
    bin_centers = np.arange(num_lags) * month
    N_taus = np.zeros(num_lags)
    retval = np.zeros(num_lags)

    for i in range(length):
        for j in range(length):

            distance = times[i] - times[j]
            index = 0

            # In order to avoid double counting into the same bin (taking the negative lag and making it positive)
            # we only calculate the ACF when the distance is positive
            while distance >= bin_centers[index]:

                if bin_centers[index] - month/2 < distance <= bin_centers[index] + month/2:
                    N_taus[index] += 1
                    retval[index] += x[i] * x[j]

                index += 1

    np.copyto(retval,  # copy to retval
              retval / N_taus,  # the values of retval / N_taus
              'same_kind',  # of the same kind
              N_taus != 0)  # only if N_taus is non-zero

    # mirror each:
    taus = bin_centers
    #    taus = np.concatenate((-1*bin_centers[::-1][:-1], bin_centers))
    #    retval = np.concatenate((retval[::-1][:-1], retval))

    return taus, retval


def autocovariance_functions(axs, nterms, narrowband_data, broadband_data):
    # find the middle points
    t = (narrowband_data[:, 1] + narrowband_data[:, 0]) / 2.0

    colors = ["#D60270", '#9B4F96', '#0038A8']

    for i, ax in enumerate(axs[:, 0]):
        diff = broadband_data[:, 2 * (i + 1)] - narrowband_data[:, 2 * (i + 1)]
        taus, retval = acf_new(t, diff - np.mean(diff))
        retval /= np.amax(retval)

        ax.plot(taus, retval, c=colors[i])
        ax.axhline(0.0, ls='--', c='black')

        ax.text(0.025, 0.05, "(" + string.ascii_lowercase[i] + ")", transform=ax.transAxes,
                size=text_size)

        if i == 0:
            ax.title.set_text('Autocovariance Functions')
            ax.set_ylabel("$R_{\Delta r_\mathrm{\infty} - \overline{\Delta r_\mathrm{\infty}}}$")
        elif i == 1:
            ax.set_ylabel("$R_{\Delta r_\mathrm{2} - \overline{\Delta r_\mathrm{2}}}$")
        else:
            ax.set_xlabel("$\\tau$ [days]")
            ax.set_ylabel("$R_{\Delta r_\mathrm{\\alpha} - \overline{\Delta r_\mathrm{\\alpha}}}$")

    return


def exponential_curve_2(x, b, tau_0):
    return b / np.exp(x / tau_0)


def characteristic_time(fig, axs, gs, narrowband_data, broadband_data, sign_figures):
    char_t_plot = fig.add_subplot(gs[:, 1])
    char_t_plot.set_title("Characteristic Time")
    char_t_plot.text(0.025, 0.025, "(d)", transform=char_t_plot.transAxes, size=text_size)
    for ax in axs[:, -1]:
        ax.remove()

    # find the middle points
    t = (narrowband_data[:, 1] + narrowband_data[:, 0]) / 2.0

    diff = broadband_data[:, 2] - narrowband_data[:, 2]
    taus, retval = acf_new(t, diff - np.mean(diff))
    retval /= np.amax(retval)

    # fit the exponential function
    x = np.array(taus[:20])
    y = np.array(retval[:20])
    x_to_plot = np.linspace(np.amin(x), np.amax(x), 10000)

    model2 = Model(exponential_curve_2)

    params2 = Parameters()
    params2.add('b', value=0.50)
    params2.add('tau_0', value=30.0)

    results2 = model2.fit(y, params=params2, x=x)
    fitted_function2 = exponential_curve_2(x_to_plot, results2.best_values['b'], results2.best_values['tau_0'])

    char_t_plot.plot(x, y, 'o-', c='C0', label="ACF")
    char_t_plot.plot(x_to_plot, fitted_function2, c="C3", lw=3, label="$\mathrm{R} = b / e^{\\tau/\\tau_0}$")
    char_t_plot.axhline(0.0, ls='--', c='black')
    char_t_plot.legend(fancybox=True, shadow=True)

    at = AnchoredText(
        "$\mathrm{b}$ = " + str(round(results2.params['b'].value, sign_figures)) + " $\pm$ " + str(
            round(results2.params['b'].stderr, sign_figures)) + "\n" +
        "$\mathrm{\\tau}_{0} = $" + str(round(results2.params['tau_0'].value, sign_figures)) + " $\pm$ " + str(
            round(results2.params['tau_0'].stderr, sign_figures)),
        prop=dict(size=text_size), frameon=True, loc='center right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    char_t_plot.add_artist(at)

    char_t_plot.set_ylabel("$R_{\Delta r_\mathrm{\infty} - \overline{\Delta r_\mathrm{\infty}}}$")
    char_t_plot.set_xlabel("$\\tau$ [days]")

    return


# Main code
PSR_name = "J1643-1224"
# PSR_name = "J1744-1134"

nterms : int = 3
sign_figures : int = 1

sns.set_style("ticks")
sns.set_context("paper") #, font_scale=2.0, rc={"lines.linewidth": 3})
#plt.rcParams.update({"text.usetex": True})

style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 21}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 24})
rc('ytick', **{'labelsize': 24})
rc('legend', **{'fontsize': 22})
rc('axes', **{'labelsize': 22, 'titlesize': 27})
text_size = 20

fig = plt.figure(figsize=(12, 8))


# Load the data_res_avg
if nterms == 2:
    gs = fig.add_gridspec(2, 2, hspace=0)
    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_2terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_2terms.txt", skiprows=1)
elif nterms == 3:
    gs = fig.add_gridspec(3, 2, hspace=0)
    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_3terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_3terms.txt", skiprows=1)

# make the plots
axs = gs.subplots(sharex=False, sharey=False)

autocovariance_functions(axs, nterms, narrowband_data, broadband_data)

characteristic_time(fig, axs, gs, narrowband_data, broadband_data, sign_figures)

plt.tight_layout()
plt.savefig("./figures/" + PSR_name + "_autocovariance.pdf")
plt.show()
