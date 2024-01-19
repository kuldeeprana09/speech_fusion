from matplotlib.font_manager import font_family_aliases
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


labels = ['BUS', 'PED', 'STR', 'QUTSTREET', 'QUTREVERB', 'QUTHOME', 'QUTCAR','QUTCAFE', 'Average(Unseen)']
# ChW_SL1_SNRi = [5.97,3.93,5.56,4.12,7.33,5.14,10.80,7.56,8.82]
# ChW_SL2_SNRi = [10.48,8.42,9.69,8.66,9.80,10.00,11.80,8.80, 9.81]
# ChW_SL3_SNRi = [10.74,8.45,10.02,9.04,10.13,10.45,12.18,8.86, 10.13]
# channelwise paper
# ChW_SL1_SNRi = [9.211,7.148,8.767,7.739,8.822,8.487,10.709,7.226, 8.5966]
#ChW_SL2_SNRi = [8.651,7.286,8.273,7.051,8.564,8.502,9.627,7.582,8.2652]
# ChW_SL3_SNRi = [9.702,7.576,9.206,7.969,9.194,9.241,10.748,7.783, 8.987]
 

#  oral defense
# channelwise
# BASE = [9.00,6.90,8.50,7.10,8.60,8.10,10.30,6.80,8.18]
# BASE_3 = [8.70,7.10,8.60,7.10,8.70,8.20,9.70,7.2,8.18]
# CW_TCN = [9.211,7.148,8.767,7.739,8.822,8.487,10.709,7.226, 8.5966]
# CWM_TCN3 = [9.702,7.576,9.206,7.969,9.194,9.241,10.748,7.783, 8.987]



# Fusion

TCN = [8.98,6.93,8.50,7.12,8.56,8.14,10.33,6.83,8.19]
MS_TCN = [9.21,7.32,8.86,7.72,9.03,8.79,10.47,7.43,8.69]
Fusion = [8.38,7.10,8.13,7.11,8.46,8.24,10.11,7.49,8.28]
MS_Fusion = [9.88,7.45,8.13,8.01,9.21,8.92,11.10,7.45,8.94]

#Mahzoon code
# TCN_SL1_SNRi = [5.97,5.51,3.93,5.56,4.12,7.33,5.14,3.83,5.19]
# TCN1_FT_SNRi = [8.38,7.10,8.12,8.45,7.10,10.10,8.23,7.49,8.12]
# TCN_SL2_SNRi = [6.21,4.31,5.82,4.72,6.02,5.79,7.47,4.43,5.69]
# ChW_SL1_SNRi = [6.21,5.77,4.15,5.82,4.74,7.71,5.49,4.23,5.60]
# TCN1_FT_SNRi = [8.38,7.10,8.12,8.45,7.10,10.10,8.23,7.49,8.12]


# Fusion paper bar plot data for -3dB
# TCN_SL1_SNRi = [8.98,6.93,8.50,7.12,8.56,8.14,10.33,6.83,8.19]
# TCN_SL2_SNRi = [9.21,7.32,8.86,7.72,9.03,8.79,10.47,7.43,8.69]

# TCN1_FT_SNRi = [8.38,7.10,8.13,7.11,8.46,8.24,10.11,7.49,8.28]
# TCN2_FT_SNRi = [9.88,7.45,8.13,8.01,9.21,8.92,11.10,7.45,8.94]


# Fusion paper bar plot data for -6dB
# TCN_SL1_SNRi = [8.07,5.76,7.72,6.01,8.17,7.10,9.72,5.53,7.31]
# TCN_SL2_SNRi = [7.22,4.97,7.11,5.62,7.75,6.41,9.11,4.61,6.70]

# TCN1_FT_SNRi = [7.28,6.06,7.05,5.97,8.05,7.16,9.40,6.44,7.40]
# TCN2_FT_SNRi = [9.21,6.42,8.63,7.29,8.88,8.16,10.87,6.19,8.28]


x = np.arange(len(labels))  # the label locations
print()
width = 0.2  # the width of the bars
fig, ax = plt.subplots(figsize=(18,10))


# rects1 = ax.bar(x, BASE, width, label='BASE', align="center")
# rects2 = ax.bar(x+width, BASE_3, width, label='BASE-3',align="center")
# rects3 = ax.bar(x+2*width, CW_TCN, width, label='CW-TCN', align="center")
# rects4 = ax.bar(x+3*width, CWM_TCN3, width, label='CWM-TCN3', align="center")

rects1 = ax.bar(x, TCN, width, label='TCN', align="center")
rects2 = ax.bar(x+width, MS_TCN, width, label='MS-TCN',align="center")
rects3 = ax.bar(x+2*width, Fusion, width, label='Fusion', align="center")
rects4 = ax.bar(x+3*width, MS_Fusion, width, label='MS-Fusion', align="center")


# rects1 = ax.bar(x, ChW_SL1_SNRi, width, color='r', hatch='O', label='CW-TCN', align="center",edgecolor='black')
# rects3 = ax.bar(x+width, ChW_SL3_SNRi, width, hatch='+', color='g', label='CWM-TCN3', align="center",edgecolor='black')
# rects3 = ax.bar(x+2*width, TCN1_FT_SNRi, width, hatch='X',  color='b', label='Fusion-Based ',align="center",edgecolor='black')
# rects4 = ax.bar(x+3*width, TCN2_FT_SNRi, width, hatch='.', color='#FEDF00', label='Multi-Slice Fusion-Based', align="center",edgecolor='black')
# rects4 = ax.bar(x+3*width, TCN2_FT_SNRi, width, hatch='\\/...', color='#FFF04D', label='Multi-Slice Fusion-Based', align="center",edgecolor='black')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SI-SNRi (dB)',fontsize=18)
ax.set_xlabel('Nosie Types', fontsize=18)
ax.set_ylim([4,11.5])
#ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

# ax.set_title('Performance Comparison for Slice-based Systems with the CahnnelWise')
# ax.set_xticks(x+3*width/2, labels)
# ax.legend()
# ax.bar_label(rects1, padding=1,fmt='%.1f')
# ax.bar_label(rects2, padding=1, fmt='%.1f')
# ax.bar_label(rects3, padding=1,fmt='%.1f')
# ax.bar_label(rects4, padding=1,fmt='%.1f')
# ax.vlines(x=[2.8], ymin=0, ymax=45, colors='black',  linestyles='dashed')
# ax.vlines(x=[7.80], ymin=0, ymax=45, colors='black',  linestyles='dashed')
#fig.tight_layout()
# fig.savefig('Multi-Slice_Fusion_SI-SNRi-3dB.jpg', dpi=900)
ax.set_xticks(x+2*width/2, labels)
ax.legend()
ax.bar_label(rects1, padding=1, fmt='%.1f')
ax.bar_label(rects2, padding=1, fmt='%.1f')
ax.bar_label(rects3, padding=1, fmt='%.1f')
ax.bar_label(rects4, padding=1, fmt='%.1f')
ax.vlines(x=[2.75], ymin=0, ymax=45, colors='black',  linestyles='dashed')
ax.vlines(x=[7.75], ymin=0, ymax=45, colors='black',  linestyles='dashed')

# fig.savefig('cwm-tcn.pdf', dpi=1200)
fig.savefig('fusion-TCN.pdf', dpi=1200)
plt.show()