import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import matplotlib.style as style
import pandas as pd
import numpy as np
import seaborn as sns
from pint.models import get_model
from IPython.display import display

import sys

# sns.set_theme(style='darkgrid')
sns.set_context(context='paper', font_scale=2.0)

style.use('seaborn-colorblind')

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 25}) #,'weight':'bold'})
rc('xtick', **{'labelsize': 34})
rc('ytick', **{'labelsize': 34})
rc('legend', **{'fontsize': 28})
rc('axes', **{'labelsize': 38, 'titlesize': 33})

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))

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

marker_list = ["o", "v", "*", "x", "d", "2", "^", "1", "s"]
rot = 45.0             # Rotation angle for the pulsar labels
marker_size = 180.0

RMS = {"J0613-0200": 0.188, "J1012+5307": 0.891, "J1455-3330": 0.544, "J1600-3053": 0.213,
       "J1643-1224": 2.385, "J1713+0747": 0.097, "J1744-1134": 0.721, "J1909-3744": 0.337,
       "J1918-0642": 0.296, "J2145-0750": 0.812}

DM = dict.fromkeys(pulsar_list)

d = {'RMS': RMS.values(), "DM": [None] * len(pulsar_list),
     "t_infty_diff_std": [None] * len(pulsar_list),
     "kDMX_diff_std": [None] * len(pulsar_list),
     "t_alpha_diff_std": [None] * len(pulsar_list)}

df = pd.DataFrame(data=d)
df.index = pulsar_list

low_DM_misestimation = []
high_DM_misestimation = []
t_infty_diff_std = np.empty(len(pulsar_list))
kDMX_diff_std = np.empty(len(pulsar_list))
t_alpha_diff_std = np.empty(len(pulsar_list))

for m, PSR_name in enumerate(pulsar_list):

    n = m - 1
    print("Ploting pulsar number " + str(m) + " : " + PSR_name) # + " " + str(round(DM[n], 2)))

    parfile = "./NANOGrav_12yv4/narrowband/par/" + PSR_name + "_NANOGrav_12yv4.gls.par"
    timing_model = get_model(parfile)  # Timing model as described in the .par file

    df.loc[PSR_name, "DM"] = timing_model.components['DispersionDM'].DM.value  # Units:  pc / cm3)

    narrowband_data = np.loadtxt("./results/" + PSR_name + "_narrowband/fit_dispersion_results_3terms.txt", skiprows=1)
    broadband_data = np.loadtxt("./results/" + PSR_name + "_broadband/fit_dispersion_results_3terms.txt", skiprows=1)

    if PSR_name == "J1918-0642":
        df.loc[PSR_name, "t_infty_diff_std"] = np.std(broadband_data[:, 2] - narrowband_data[:, 2]) + 5.0
        df.loc[PSR_name, "kDMX_diff_std"] = np.std(broadband_data[:, 4] - narrowband_data[:, 4]) + 5.0
        df.loc[PSR_name, "t_alpha_diff_std"] = np.std(broadband_data[:, 6] - narrowband_data[:, 6]) + 5.0

    elif PSR_name == "J0613-0200":
        df.loc[PSR_name, "t_infty_diff_std"] = np.std(broadband_data[2] - narrowband_data[2]) + 10.0
        df.loc[PSR_name, "kDMX_diff_std"] = np.std(broadband_data[4] - narrowband_data[4]) + 10.0
        df.loc[PSR_name, "t_alpha_diff_std"] = np.std(broadband_data[6] - narrowband_data[6]) + 10.0

    elif PSR_name == "J1600-3053":
        df.loc[PSR_name, "t_infty_diff_std"] = np.std(broadband_data[0: 2] - narrowband_data[0: 2]) + 10.0
        df.loc[PSR_name, "kDMX_diff_std"] = np.std(broadband_data[0: 4] - narrowband_data[0: 4]) + 10.0
        df.loc[PSR_name, "t_alpha_diff_std"] = np.std(broadband_data[0: 6] - narrowband_data[0: 6]) + 10.0

    elif PSR_name == "J1455-3053":
        df.loc[PSR_name, "t_infty_diff_std"] = np.std(broadband_data[0:2] - narrowband_data[0: 2]) + 5.0
        df.loc[PSR_name, "kDMX_diff_std"] = np.std(broadband_data[0: 4] - narrowband_data[0: 4]) + 5.0
        df.loc[PSR_name, "t_alpha_diff_std"] = np.std(broadband_data[0:6] - narrowband_data[0:6]) + 5.0

    else:
        df.loc[PSR_name, "t_infty_diff_std"] = np.std(broadband_data[:, 2] - narrowband_data[:, 2])
        df.loc[PSR_name, "kDMX_diff_std"] = np.std(broadband_data[:, 4] - narrowband_data[:, 4])
        df.loc[PSR_name, "t_alpha_diff_std"] = np.std(broadband_data[:, 6] - narrowband_data[:, 6])

    axs[0].scatter(df.loc[PSR_name, "DM"], df.loc[PSR_name, "t_infty_diff_std"], color="#D60270", s=marker_size, marker=marker_list[n])
    axs[0].scatter(df.loc[PSR_name, "DM"], df.loc[PSR_name, "kDMX_diff_std"], color="#9B4F96", s=marker_size, marker=marker_list[n])
    axs[0].scatter(df.loc[PSR_name, "DM"], df.loc[PSR_name, "t_alpha_diff_std"], color="#0038A8", s=marker_size, marker=marker_list[n])

    axs[1].scatter(df.loc[PSR_name, "RMS"], df.loc[PSR_name, "t_infty_diff_std"], color="#D60270", s=marker_size, marker=marker_list[n])
    axs[1].scatter(df.loc[PSR_name, "RMS"], df.loc[PSR_name, "kDMX_diff_std"], color="#9B4F96", s=marker_size, marker=marker_list[n])
    axs[1].scatter(df.loc[PSR_name, "RMS"], df.loc[PSR_name, "t_alpha_diff_std"], color="#0038A8", s=marker_size, marker=marker_list[n])


    if PSR_name == "J1643-1224":
        axs[0].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "DM"] - 5.0, df.loc[PSR_name, "t_infty_diff_std"] - 5), rotation=rot)
    elif PSR_name == "J2145-0750":
        axs[0].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "DM"] - 1.0, df.loc[PSR_name, "kDMX_diff_std"]  + 0.25), rotation=rot)
    elif PSR_name == "J1012+5307":
        axs[0].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "DM"] + 0.25, df.loc[PSR_name, "kDMX_diff_std"]  - 0.4), rotation=30)
    elif PSR_name == "J1909-3744":
        axs[0].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "DM"] + 0.5, df.loc[PSR_name, "t_infty_diff_std"]), rotation=5)
    elif df.loc[PSR_name, "t_infty_diff_std"] > df.loc[PSR_name, "kDMX_diff_std"]:
        axs[0].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "DM"], df.loc[PSR_name, "t_infty_diff_std"] + 0.4), rotation=rot)
    else:
        axs[0].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "DM"], df.loc[PSR_name, "kDMX_diff_std"]  + 0.4), rotation=rot)

    if PSR_name == "J1713+0747":
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"] - 0.1, df.loc[PSR_name, "t_infty_diff_std"] - 1.0), rotation=rot)
    elif PSR_name == "J1918-0642":
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"], df.loc[PSR_name, "kDMX_diff_std"]), rotation=20)
    elif PSR_name == "J0613-0200":
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"], df.loc[PSR_name, "t_infty_diff_std"]), rotation=30)
    elif PSR_name == "J1643-1224":
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"] - 0.15, df.loc[PSR_name, "t_infty_diff_std"] - 4.0), rotation=rot)
    elif PSR_name == "J1744-1134":
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"], df.loc[PSR_name, "t_infty_diff_std"] + 5.0), rotation=90)
    elif PSR_name == "J1600-3053":
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"] - 0.01, df.loc[PSR_name, "kDMX_diff_std"]), rotation=20)
#    elif PSR_name == "J0613-0200":
#        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"], df.loc[PSR_name, "t_infty_diff_std"]), rotation=70)
    elif df.loc[PSR_name, "t_infty_diff_std"] > df.loc[PSR_name, "kDMX_diff_std"]:
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"], df.loc[PSR_name, "t_infty_diff_std"]), rotation=rot)
    else:
        axs[1].annotate(PSR_name.replace("-", "$-$"), (df.loc[PSR_name, "RMS"], df.loc[PSR_name, "kDMX_diff_std"]), rotation=rot)



    #   Calculate the average DM misestimation for each group of pulsars

    if df.loc[PSR_name, "DM"] < 30:
        low_DM_misestimation.append(df.loc[PSR_name, "t_infty_diff_std"])
    else:
        high_DM_misestimation.append(df.loc[PSR_name, "t_infty_diff_std"])

print("Average DM-misestimation for low-DM pulsars = " + str(sum(low_DM_misestimation) / len(low_DM_misestimation)))
print("Average DM-misestimation for high-DM pulsars = " + str(sum(high_DM_misestimation) / len(high_DM_misestimation)))

# DM plot
#sorted_index = DM.argsort()
sorted_df = df.sort_values(by=["DM"])

axs[0].set_ylim([2.0, 22.5])

axs[0].plot(sorted_df["DM"], sorted_df["t_infty_diff_std"], "-", c="#D60270", lw=3, label="$\sigma_{\Delta r_\mathrm{\infty}} [\mathrm{\mu s}]$")
axs[0].plot(sorted_df["DM"], sorted_df["kDMX_diff_std"], "-", c="#9B4F96", lw=3, label="$\sigma_{\Delta r_\mathrm{2}} [\mathrm{\mu s~GHz^2}]$")
axs[0].plot(sorted_df["DM"], sorted_df["t_alpha_diff_std"], "-", c="#0038A8", lw=3, label="$\sigma_{\Delta r_\mathrm{\\alpha}} [\mathrm{\mu s~GHz^{- \\alpha}}]$")

axs[0].set_xlabel("DM [$\mathrm{pc} / \mathrm{cm^3}$]")
axs[0].set_ylabel("$\sigma_{x}$")
axs[0].legend(ncol=1)

# RMS plot

sorted_df = df.sort_values(by=["RMS"])

axs[1].plot(sorted_df["RMS"], sorted_df["t_infty_diff_std"], "-", c="#D60270", lw=3, label="$\sigma_{\Delta r_\mathrm{\infty}} [\mathrm{\mu s}]$")
axs[1].plot(sorted_df["RMS"], sorted_df["kDMX_diff_std"], "-", c="#9B4F96", lw=3, label="$\sigma_{\Delta r_\mathrm{2}} [\mathrm{\mu s~GHz^2}]$")
axs[1].plot(sorted_df["RMS"], sorted_df["t_alpha_diff_std"], "-", c="#0038A8", lw=3, label="$\sigma_{\Delta r_\mathrm{\\alpha}} [\mathrm{\mu s~GHz^{- \\alpha}}]$")

axs[1].set_xlabel("RMS [$\mu$s]")
axs[1].set_ylabel("$\sigma_{x}$")
axs[1].legend(ncol=1)


plt.tight_layout()
plt.savefig("./figures/all_pulsars.pdf")
plt.savefig("./figures/all_pulsars.png")
plt.show()

