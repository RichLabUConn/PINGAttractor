import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm2
from statsmodels.formula.api import ols


#%% Read desired data
datalist=[]
datalist_names=[]
data_total=[]
for E_GABA in [75, 73]:
    for NoiseAmp in [0, 2000]:
        csvstr='GammaQuantification%d/Stats_GammaFeaturesIFRH_EGABA%d_NoiseAmp%d_Iext30_v2.csv' %(E_GABA, E_GABA, NoiseAmp)
        temp=np.genfromtxt(csvstr, delimiter=',')
        temp2=np.append(temp, np.ones([len(temp),1])*-E_GABA,1)
        temp3=np.append(temp2, np.ones([len(temp2),1])*NoiseAmp,1)
        datalist.append(temp3)
        datalist_names.append(csvstr)
        if len(datalist)==1:
            data_total=datalist[-1]
        else:
            data_total=np.concatenate((data_total,datalist[-1]),axis=0)
            
#%% Create Data Frame
df=pd.DataFrame(data_total)
df=df.rename(columns={0:'Number', 1:'Non-PING', 2:'Start', 3:'Stop', 4:'Duration', 5:'PeakFreq', 6:'EGABA', 7:'NoiseAmp'})

# Extract only the columns needed for ANOVA
anova_df = pd.DataFrame({
    "EventDuration": df.iloc[:, 4],
    "PeakFrequency": df.iloc[:, 5],
    "E_GABA": df.iloc[:, 6],
    "NoiseAmp": df.iloc[:, 7]
})

# Treat the experimental factors as categorical variables
anova_df["E_GABA"] = anova_df["E_GABA"].astype("category")
anova_df["NoiseAmp"] = anova_df["NoiseAmp"].astype("category")

# Remove rows containing missing values
anova_df = anova_df.dropna()



#%% Two-way ANOVA: Event Duration

model_duration = ols(
    "EventDuration ~ C(E_GABA) * C(NoiseAmp)",
    data=anova_df
).fit()

anova_duration = sm2.stats.anova_lm(model_duration, typ=2)

print("\nEvent Duration ANOVA")
print(anova_duration)

# residuals=model_duration.resid
# plt.hist(residuals, bins=30)
# plt.xlabel("Residual")
# plt.ylabel("Count")
# plt.show()

# import scipy.stats as stats

# stats.probplot(residuals, dist="norm", plot=plt)
# plt.show()

#%% Two-way ANOVA: Peak Frequency

model_frequency = ols(
    "PeakFrequency ~ C(E_GABA) * C(NoiseAmp)",
    data=anova_df
).fit()

anova_frequency = sm2.stats.anova_lm(model_frequency, typ=2)

print("\nPeak Frequency ANOVA")
print(anova_frequency)

# # ==============================================================
# # Assumption checks for Peak Frequency ANOVA
# # ==============================================================

# # import matplotlib.pyplot as plt
# # import seaborn as sns
# import scipy.stats as stats
# from scipy.stats import levene

# # --------------------------------------------------------------
# # Extract residuals
# # --------------------------------------------------------------
# residuals = model_frequency.resid

# # --------------------------------------------------------------
# # Histogram of residuals
# # --------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(4, 3))

# sns.histplot(
#     residuals,
#     bins=30,
#     kde=True,
#     ax=ax
# )

# ax.set_xlabel('Residual')
# ax.set_ylabel('Count')
# ax.set_title('Peak Frequency Residuals')

# plt.tight_layout()
# plt.show()

# # --------------------------------------------------------------
# # Q-Q plot
# # --------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(4, 4))

# stats.probplot(
#     residuals,
#     dist='norm',
#     plot=ax
# )

# ax.set_title('Peak Frequency Residual Q-Q Plot')

# plt.tight_layout()
# plt.show()

# # --------------------------------------------------------------
# # Levene's test for equal variances
# # --------------------------------------------------------------
# groups = [
#     group['PeakFrequency'].values
#     for _, group in anova_df.groupby(['E_GABA', 'NoiseAmp'])
# ]

# levene_stat, levene_p = levene(*groups)

# print('\nLevene Test')
# print('------------')
# print(f'Statistic = {levene_stat:.4f}')
# print(f'p-value   = {levene_p:.4g}')

# if levene_p < 0.05:
#     print('Evidence for unequal variances between groups.')
# else:
#     print('No evidence for unequal variances between groups.')

# # --------------------------------------------------------------
# # Group sample sizes
# # --------------------------------------------------------------
# print('\nGroup Sizes')
# print('-----------')
# print(
#     anova_df.groupby(['E_GABA', 'NoiseAmp'])
#     .size()
# )

#%% Mann-Whitney test
# mw_duration=[]
# mw_peakfreq=[]
# for i in range(2):
#     nonoise_duration=datalist[i*2][:,4]
#     noise_duration=datalist[i*2+1][:,4]
#     nonoise_peakfreq=datalist[i*2][:,5]
#     noise_peakfreq=datalist[i*2+1][:,5]
    
#     U,p=stats.mannwhitneyu(nonoise_duration,noise_duration)
#     mw_duration.append(p)
    
#     U,p=stats.mannwhitneyu(nonoise_peakfreq,noise_peakfreq)
#     mw_peakfreq.append(p)



#%% Calculate means
duration_means=df.groupby(['EGABA', 'NoiseAmp'])['Duration'].mean()
peakfreq_means=df.groupby(['EGABA', 'NoiseAmp'])['PeakFreq'].mean()
         
#%% Plot


plt.rcParams['font.size'] = 8

# ------------------------------------------------------------------
# Create a combined condition variable
# ------------------------------------------------------------------
df['Condition'] = (
    df['EGABA'].astype(int).astype(str) + '_' +
    df['NoiseAmp'].astype(int).astype(str)
)

# Define order explicitly
condition_order = [
    '-75_0',
    '-75_2000',
    '-73_0',
    '-73_2000'
]

# Four colors from the jet colormap
jet_colors = plt.cm.jet(np.linspace(0, 1, 4))

palette = {
    '-75_0'    : jet_colors[0],
    '-75_2000' : jet_colors[1],
    '-73_0'    : jet_colors[2],
    '-73_2000' : jet_colors[3]
}

# ==============================================================
# Duration plot
# ==============================================================

fig, ax1 = plt.subplots()
fig.set_size_inches(4.0, 2)

ax = sns.violinplot(
    x='Condition',
    y='Duration',
    data=df,
    order=condition_order,
    palette=palette,
    inner=None,
    ax=ax1
)

plt.scatter(
    x=np.arange(4),
    y=duration_means,
    c='k',
    s=15,
    zorder=10
)


ax.set_xlabel(r'$E_{\mathrm{GABA}}$ (mV), $SD_V$ (mV)')
ax.set_xticklabels([
    '-75, 0.000',
    '-75, 1.087',
    '-73, 0.000',
    '-73, 1.087'
])

ax.xaxis.label.set_alpha(0)
for tick in ax.get_xticklabels():
    tick.set_alpha(0)

ax.set_ylabel('Gamma event duration (s)')

ax.spines[['right', 'top']].set_visible(False)

filenamestr = 'PaperFigures/Figure4/GammaDuration.png'
plt.savefig(
    filenamestr,
    dpi=600,
    format='png',
    bbox_inches='tight'
)

# ==============================================================
# Peak frequency plot
# ==============================================================

fig, ax1 = plt.subplots()
fig.set_size_inches(4.0, 2)

ax = sns.violinplot(
    x='Condition',
    y='PeakFreq',
    data=df,
    order=condition_order,
    palette=palette,
    inner=None,
    ax=ax1
)

plt.scatter(
    x=np.arange(4),
    y=peakfreq_means,
    c='k',
    s=15,
    zorder=10
)

ax.set_xlabel(r'$E_{\mathrm{GABA}}$ (mV)' + ', ' + r'$SD_V$ (mV)')
ax.set_xticklabels([
    '-75, 0.000',
    '-75, 1.087',
    '-73, 0.000',
    '-73, 1.087'
])

ax.set_ylabel('Peak Frequency (Hz)')
ax.spines[['right', 'top']].set_visible(False)






filenamestr = 'PaperFigures/Figure4/GammaPeakFreq.png'
plt.savefig(
    filenamestr,
    dpi=600,
    format='png',
    bbox_inches='tight'
)