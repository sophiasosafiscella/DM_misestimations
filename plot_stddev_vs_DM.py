import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import matplotlib.style as style

import numpy as np
import seaborn as sns
from pint.models import get_model

# sns.set_theme(style='darkgrid')
sns.set_context(context='paper', font_scale=1.5)

style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 21}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 24})
rc('ytick', **{'labelsize': 24})
rc('legend', **{'fontsize': 24})
rc('axes', **{'labelsize': 22, 'titlesize': 27})

fig = plt.figure(figsize=(12, 8))

pulsar_list = ["J0613-0200",
               "J1012+5307",  # This one is being mean to me :(
               "J1455-3330",  # This one is being mean to me :(
               "J1600-3053",
               "J1643-1224",
               "J1713+0747",  # Oh, this is a new one!
               "J1744-1134",
               "J1909-3744",
               "J1918-0642",  # This one is being mean to me :(
               "J2145-0750"]

marker_list = ["o", "v", "*", "x", "d", "s", "^", "1", "2"]

DM = np.empty(len(pulsar_list))
low_DM_misestimation = []
high_DM_misestimation = []
t_infty_diff_std = np.empty(len(pulsar_list))
kDMX_diff_std = np.empty(len(pulsar_list))
t_alpha_diff_std = np.empty(len(pulsar_list))

for m, PSR_name in enumerate(pulsar_list):

    n = m - 1
    print("Ploting pulsar number " + str(m) + " : " + PSR_name) # + " " + str(round(DM[n], 2)))

    parfile = "../dm_misestimations/NANOGrav_12yv4/narrowband/par/" + PSR_name + "_NANOGrav_12yv4.gls.par"
    timing_model = get_model(parfile)  # Timing model as described in the .par file

    DM[n] = timing_model.components['DispersionDM'].DM.value  # Units:  pc / cm3)

    narrowband_data = np.loadtxt("./" + PSR_name + "_narrowband/fit_dispersion_results_3terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./" + PSR_name + "_broadband/fit_dispersion_results_3terms.txt", skiprows=1)

    if PSR_name == "J1918-06422":
        t_infty_diff_std[n] = np.std(broadband_data[2] - narrowband_data[2])
        kDMX_diff_std[n] = np.std(broadband_data[4] - narrowband_data[4])
        t_alpha_diff_std[n] = np.std(broadband_data[6] - narrowband_data[6])

        plt.scatter(DM[n], t_infty_diff_std[n], color="C0", s=60, marker=marker_list[n], label=PSR_name)
        plt.scatter(DM[n], kDMX_diff_std[n], color="C1", s=60, marker=marker_list[n])
        plt.scatter(DM[n], t_alpha_diff_std[n], color="C2", s=60, marker=marker_list[n])
    else:
        t_infty_diff_std[n] = np.std(broadband_data[:, 2] - narrowband_data[:, 2])
        kDMX_diff_std[n] = np.std(broadband_data[:, 4] - narrowband_data[:, 4])
        t_alpha_diff_std[n] = np.std(broadband_data[:, 6] - narrowband_data[:, 6])

        plt.scatter(DM[n], t_infty_diff_std[n], color="C0", s=60, marker=marker_list[n], label=PSR_name)
        plt.scatter(DM[n], kDMX_diff_std[n], color="C1", s=60, marker=marker_list[n])
        plt.scatter(DM[n], t_alpha_diff_std[n], color="C2", s=60, marker=marker_list[n])

    #   Calculate the average DM misestimation for each group of pulsars

    if DM[n] < 30:
        low_DM_misestimation.append(t_infty_diff_std[n])
    else:
        high_DM_misestimation.append(t_infty_diff_std[n])

print("Average DM-misestimation for low-DM pulsars = " + str(sum(low_DM_misestimation) / len(low_DM_misestimation)))
print("Average DM-misestimation for high-DM pulsars = " + str(sum(high_DM_misestimation) / len(high_DM_misestimation)))

sorted_index = DM.argsort()

plt.plot(DM[sorted_index], t_infty_diff_std[sorted_index], "-", c="C0", label="$\sigma_{\Delta r_\mathrm{\infty}}$")
plt.plot(DM[sorted_index], kDMX_diff_std[sorted_index], "-", c="C1", label="$\sigma_{\Delta r_\mathrm{2}}$")
plt.plot(DM[sorted_index], t_alpha_diff_std[sorted_index], "-", c="C2", label="$\sigma_{\Delta r_\mathrm{\\alpha}}$")

plt.xlabel("DM [$\mathrm{pc} / \mathrm{cm^3}$]")
plt.ylabel("$\sigma$ [$\mathrm{\mu s}$]")
plt.legend(ncol=3)
plt.tight_layout()
#plt.savefig("all_pulsars.pdf")
plt.show()
