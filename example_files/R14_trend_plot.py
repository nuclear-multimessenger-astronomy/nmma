import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn
c = seaborn.color_palette('colorblind')

fig_width_pt = 750.  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1. / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.) / 2.  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 0.9 * fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'pdf',
          'axes.labelsize': 18,
          'legend.fontsize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True,
          'font.family': 'Times New Roman',
          'figure.figsize': fig_size}
matplotlib.rcParams.update(params)

# user specify input
R14_true = 11.55
label = 'ZTF'

data_GWEM = pd.read_csv('GW_EM_R14trend_{0}.dat'.format(label), header=0, delimiter=' ')

data_GW = pd.read_csv('GW_R14trend.dat', header=0, delimiter=' ')

fig = plt.figure()
fig.suptitle("Constrain EoS using EM + GW ", fontname="Times New Roman Bold")
ax1 = plt.subplot2grid((4,5),(0,0), rowspan = 3, colspan = 4)
ax2 = plt.subplot2grid((4,5),(3,0), colspan = 4, sharex=ax1)
ax1.set_xlim([0.5, len(data_GWEM) + 0.5])
ax1.set_ylabel(r'$R_{1.4} \ [{\rm km}]$')
ax2.set_ylabel(r'$\delta R_{1.4} / R_{1.4} \ [\%]$')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set_xticks(np.arange(1, len(data_GWEM) + 1, 2))
ax2.set_xlabel('Events')

axis_GW = np.arange(1, len(data_GW) + 1)
axis_GWEM = np.arange(1, len(data_GWEM) + 1)
ax1.errorbar(axis_GW, data_GW.R14_med, yerr=[data_GW.R14_lowerr, data_GW.R14_uperr], label='GW', color=c[3], fmt='o', capsize=5)
ax1.errorbar(axis_GWEM, data_GWEM.R14_med, yerr=[data_GWEM.R14_lowerr, data_GWEM.R14_uperr], label='GW+EM', color=c[0], fmt='o', capsize=5)
ax1.axhline(R14_true, linestyle='--', color=c[1], label='Injected value')
ax1.legend()

GW_mean_error = np.mean([data_GW.R14_lowerr, data_GW.R14_uperr], axis=0)
GWEM_mean_error = np.mean([data_GWEM.R14_lowerr, data_GWEM.R14_uperr], axis=0)
ax2.plot(axis_GW, GW_mean_error / data_GW.R14_med * 100, color=c[3], marker='o')
ax2.plot(axis_GWEM, GWEM_mean_error / data_GWEM.R14_med * 100, color=c[0], marker='o')
ax2.set_yscale('log')
ax2.axhline(10, color='grey' ,linestyle='--', alpha=0.5)
ax2.axhline(5, color='grey' ,linestyle='--', alpha=0.5)
ax2.axhline(1, color='grey' ,linestyle='--', alpha=0.5)

fig.tight_layout()
fig.subplots_adjust(hspace=0.1)
plt.savefig('R14_trend_GW_EM_{0}.pdf'.format(label), bbox_inches='tight')
