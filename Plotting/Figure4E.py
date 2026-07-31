#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
# import os
import csv
# import time
from scipy.signal import find_peaks
# import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.formula.api import glm
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import levene


#%% Noise Values
noiseAmps=[0, 1, 20, 70, 200, 600, 1000, 2000, 4000, 6000, 8000]
extralabel='all'

noiseAmps=[noiseAmps[x] for x in [0,7]]
extralabel='Fig4E'


Iextnum=11
Iextmin=.25
Iextmax=.5
Istep=0.025

repetitions=100

#%% Intitialize

syncpercent=np.zeros((Iextnum,len(noiseAmps)))


E_GABA_list=[75, 73]
NoiseAmp_list=[0, 2000]
Iext_list=[25, 27.5, 30, 32.5, 35]
Iext_plot=[x/100 for x in Iext_list]

syncpercent_USE=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]




#%% ANOVA

# Read desired data
datalist=[]
datalist_names=[]
data_total=[]
for E_GABA in [75, 73]:
    for NoiseAmp in [0, 2000]:
        csvstr='Figure4ANOVARawData/GammaEventRate_SyncPercent_EGABA%d_NoiseAmp%d.csv' %(E_GABA, NoiseAmp)
        temp=np.genfromtxt(csvstr, delimiter=',')
        datalist.append(temp)
        datalist_names.append(csvstr)
        if len(datalist)==1:
            data_total=datalist[-1]
        else:
            data_total=np.concatenate((data_total,datalist[-1]),axis=0)
            
#%% Create Data Frame
df=pd.DataFrame(data_total)
df=df.rename(columns={0:'EGABA', 1:'NoiseAmp', 2:'Iext', 3:'Number', 4:'GammaRate', 5:'ConsistentAndLasting'})

# Treat the experimental factors as categorical variables
df["EGABA"] = df["EGABA"].astype("category")
df["NoiseAmp"] = df["NoiseAmp"].astype("category")

# Remove rows containing missing values
df = df.dropna()


# ============================================================
# Assign readable column names
# ============================================================

df.columns = [
    "EGABA",
    "NoiseAmp",
    "Iext",
    "Number",
    "GammaRate",
    "ConsistentAndLasting"
]

# ============================================================
# Function to run ANOVA at each Iext and save full statistics
# ============================================================

def run_anova_per_iext(df, response_variable, output_csv):

    summary_rows = []   # Contains only p-values for console output
    full_rows = []      # Contains full ANOVA statistics for CSV output

    for iext in sorted(df["Iext"].unique()):

        subset = df[df["Iext"] == iext]

        # Need enough observations to fit the model
        if len(subset) < 4:
            continue

        model = ols(
            f"{response_variable} ~ C(EGABA) * C(NoiseAmp)",
            data=subset
        ).fit()

        anova_table = anova_lm(model, typ=2)

        # ----------------------------------------------------
        # Save full ANOVA statistics
        # ----------------------------------------------------
        for term in anova_table.index:

            full_rows.append({
                "Iext": iext,
                "Term": term,
                "sum_sq": anova_table.loc[term, "sum_sq"],
                "df": anova_table.loc[term, "df"],
                "F": anova_table.loc[term, "F"],
                "p_value": anova_table.loc[term, "PR(>F)"]
            })

        # ----------------------------------------------------
        # Save p-values only for concise display
        # ----------------------------------------------------
        summary_rows.append({
            "Iext": iext,
            "EGABA_p": anova_table.loc["C(EGABA)", "PR(>F)"],
            "NoiseAmp_p": anova_table.loc["C(NoiseAmp)", "PR(>F)"],
            "Interaction_p": anova_table.loc["C(EGABA):C(NoiseAmp)", "PR(>F)"]
        })

    # Create dataframes
    summary_df = pd.DataFrame(summary_rows)
    full_df = pd.DataFrame(full_rows)

    # Save full ANOVA statistics
    full_df.to_csv(output_csv, index=False)

    return summary_df

# ============================================================
# Logistic regression for binary response variables
# ============================================================

def run_logistic_per_iext(df, response_variable, output_csv):

    summary_rows = []
    full_rows = []

    for iext in sorted(df["Iext"].unique()):

        subset = df[df["Iext"] == iext]

        # Cannot fit logistic regression if response is constant
        if subset[response_variable].nunique() <= 1:

            summary_rows.append({
                "Iext": iext,
                "EGABA_p": np.nan,
                "NoiseAmp_p": np.nan,
                "Interaction_p": np.nan
            })

            continue

        model = glm(
            f"{response_variable} ~ C(EGABA) * C(NoiseAmp)",
            data=subset,
            family=sm.families.Binomial()
        ).fit()

        # Save full model statistics
        for term in model.params.index:

            full_rows.append({
                "Iext": iext,
                "Term": term,
                "Coefficient": model.params[term],
                "StdErr": model.bse[term],
                "z_value": model.tvalues[term],
                "p_value": model.pvalues[term]
            })

        # Extract term p-values
        summary_rows.append({
            "Iext": iext,
            "EGABA_p":
                model.pvalues.get("C(EGABA)[T.75.0]", np.nan),
            "NoiseAmp_p":
                model.pvalues.get("C(NoiseAmp)[T.2000.0]", np.nan),
            "Interaction_p":
                model.pvalues.get(
                    "C(EGABA)[T.75.0]:C(NoiseAmp)[T.2000.0]",
                    np.nan
                )
        })

    summary_df = pd.DataFrame(summary_rows)
    full_df = pd.DataFrame(full_rows)

    full_df.to_csv(output_csv, index=False)

    return summary_df

# ============================================================
# GammaRate ANOVA
# ============================================================

gamma_pvalues = run_anova_per_iext(
    df,
    response_variable="GammaRate",
    output_csv="Figure4ANOVARawData/GammaRate_ANOVA_full_stats.csv"
)

# ============================================================
# Generate significance annotations automatically
# ============================================================

gamma_annotations = {}

for _, row in gamma_pvalues.iterrows():

    annotation = ""

    if row["EGABA_p"] < 0.05:
        annotation += "* "

    if row["NoiseAmp_p"] < 0.05:
        annotation += "# "

    if row["Interaction_p"] < 0.05:
        annotation += "\u2020"

    i = len(gamma_annotations)
    gamma_annotations[Iext_plot[i]] = annotation
    
print("\nGammaRate significance annotations:")
for iext, annotation in gamma_annotations.items():
    print(f"Iext={iext}: {annotation}")

print("\nGammaRate ANOVA p-values")
print(gamma_pvalues.to_string(index=False))

# ============================================================
# ConsistentAndLasting LOGISTIC
# ============================================================

consistent_pvalues = run_logistic_per_iext(
    df,
    response_variable="ConsistentAndLasting",
    output_csv="Figure4ANOVARawData/ConsistentAndLasting_Logistic_full_stats.csv"
)

# ============================================================
# Generate significance annotations automatically
# ============================================================

consistent_annotations = {}

for _, row in consistent_pvalues.iterrows():

    annotation = ""

    if pd.notna(row["EGABA_p"]) and row["EGABA_p"] < 0.05:
        annotation += "* "

    if pd.notna(row["NoiseAmp_p"]) and row["NoiseAmp_p"] < 0.05:
        annotation += "# "

    if pd.notna(row["Interaction_p"]) and row["Interaction_p"] < 0.05:
        annotation += "\u2020"

    i = len(consistent_annotations)
    consistent_annotations[Iext_plot[i]] = annotation

print("\nConsistentAndLasting significance annotations:")
for iext, annotation in consistent_annotations.items():
    print(f"Iext={iext}: {annotation}")

print("\nConsistentAndLasting logistic regression p-values")
print(consistent_pvalues.to_string(index=False))


#%% ANOVA checks


# ============================================================
# Assign readable column names
# ============================================================

df.columns = [
    "EGABA",
    "NoiseAmp",
    "Iext",
    "Number",
    "GammaRate",
    "ConsistentAndLasting"
]

# ============================================================
# ANOVA assumption checks
# ============================================================

def check_anova_assumptions(df, response_var):

    print("\n")
    print("=" * 80)
    print(f"ASSUMPTION CHECKS: {response_var}")
    print("=" * 80)

    for iext in sorted(df["Iext"].unique()):

        subset = df[df["Iext"] == iext]

        print("\n" + "-" * 80)
        print(f"Iext = {iext}")
        print("-" * 80)

        # ====================================================
        # Design matrix check
        # ====================================================

        crosstab = pd.crosstab(
            subset["EGABA"],
            subset["NoiseAmp"]
        )

        print("\nCell counts:")
        print(crosstab)

        # Number of levels present
        n_egaba = subset["EGABA"].nunique()
        n_noise = subset["NoiseAmp"].nunique()

        print(
            f"\nLevels present: "
            f"EGABA={n_egaba}, "
            f"NoiseAmp={n_noise}"
        )

        # ====================================================
        # Response variability
        # ====================================================

        response = subset[response_var]

        print(
            f"\nResponse statistics:"
            f"\n  N observations = {len(response)}"
            f"\n  Unique values   = {response.nunique()}"
            f"\n  Variance        = {response.var()}"
        )

        # Skip tests if response is constant
        if response.nunique() <= 1:

            print(
                "\nWARNING: Response variable "
                "is constant. ANOVA cannot be performed."
            )

            continue

        # ====================================================
        # Build ANOVA model
        # ====================================================

        try:

            model = ols(
                f"{response_var} ~ C(EGABA) * C(NoiseAmp)",
                data=subset
            ).fit()

        except Exception as e:

            print("\nModel fitting failed:")
            print(e)

            continue

        # ====================================================
        # Shapiro-Wilk test on residuals
        # ====================================================

        residuals = model.resid

        if len(residuals) >= 3:

            shapiro_stat, shapiro_p = shapiro(residuals)

            print(
                "\nShapiro-Wilk test (residual normality)"
                f"\n  W = {shapiro_stat:.4f}"
                f"\n  p = {shapiro_p:.4g}"
            )

            if shapiro_p < 0.05:
                print(
                    "  WARNING: Residuals deviate "
                    "from normality."
                )

        # ====================================================
        # Levene's test
        # ====================================================

        groups = []

        for (_, _), group_df in subset.groupby(
            ["EGABA", "NoiseAmp"]
        ):

            if len(group_df) > 1:
                groups.append(group_df[response_var].values)

        if len(groups) == 4:

            lev_stat, lev_p = levene(*groups)

            print(
                "\nLevene test (equal variances)"
                f"\n  Statistic = {lev_stat:.4f}"
                f"\n  p = {lev_p:.4g}"
            )

            if lev_p < 0.05:
                print(
                    "  WARNING: Evidence for "
                    "unequal variances."
                )

        else:

            print(
                "\nLevene test not performed:"
                "\nNot all four groups were available."
            )

        # ====================================================
        # Replication check
        # ====================================================

        min_cell_size = subset.groupby(
            ["EGABA", "NoiseAmp"]
        ).size().min()

        print(
            f"\nMinimum cell size = {min_cell_size}"
        )

        if min_cell_size < 2:

            print(
                "WARNING: One or more cells contain "
                "fewer than 2 observations."
            )



# ============================================================
# Run checks
# ============================================================

# check_anova_assumptions(df, "GammaRate")
# check_anova_assumptions(df, "ConsistentAndLasting")




#%% Figure 4E Bottom
# Import and store data for plotting EGABA 75
for i, noiseamp in enumerate(noiseAmps): 
    
    print(noiseamp)
    str1=''
         
    stats_csv=np.genfromtxt(str1, delimiter=',')
    
    str2=''
         
    burstfreq_csv=np.genfromtxt(str2, delimiter=',')

    syncpercent[:,i]=stats_csv[:,49]

syncpercent_USE[0]=syncpercent[:len(Iext_list),0].tolist()
syncpercent_USE[1]=syncpercent[:len(Iext_list),1].tolist()

# Import and store data for plotting EGABA 73
Iextnum=5
syncpercent=np.zeros((Iextnum,len(noiseAmps)))

for i, noiseamp in enumerate(noiseAmps): 
    
    print(noiseamp)
    str1=''
         
    stats_csv=np.genfromtxt(str1, delimiter=',')
    
    # str2='/Users/sbr23005/Library/CloudStorage/OneDrive-UniversityofConnecticut/101) SBR_ResearchStorage - Documents/PostdocPapers/1) Steve Paper/1) Redoing simulations/3a) New initial conditions, EGABA -73/AverageOverRepetitionsPlots/NewBurstFreq_VaryIext_Repetitions%d_NoiseAmp%d.csv' \
    #      % (repetitions, noiseamp)
         
    # burstfreq_csv=np.genfromtxt(str2, delimiter=',')

    syncpercent[:,i]=stats_csv[:,49]

syncpercent_USE[2]=syncpercent[:len(Iext_list),0].tolist()
syncpercent_USE[3]=syncpercent[:len(Iext_list),1].tolist()



#Plot
NoiseAmp_to_VoltageSD=dict([(0,0),(1,0.023),(20,0.108), (70, 0.199), (200, 0.325), (600, 0.581), (1000, 0.750), (2000,1.087), (4000,1.439), (6000, 1.842), (8000, 2.234)])


cmap=cm.get_cmap('jet')

plt.rcParams['font.size']=8
fig, ax=plt.subplots()
fig.set_size_inches(2.5,2)


listindex=0
for E_GABA in E_GABA_list:
    for NoiseAmp in NoiseAmp_list:
        str1='-%d, %1.3f' % (E_GABA, NoiseAmp_to_VoltageSD[NoiseAmp])
        ax.plot(Iext_plot, syncpercent_USE[listindex], color=cmap(listindex/(len(syncpercent_USE)-1)), label=str1, lw=2)
        # ax.fill_between(Iext_list, nonPING_mean[listindex, :len(Iext_list)]+nonPING_std[listindex, :len(Iext_list)],
        #                 nonPING_mean[listindex, :len(Iext_list)]-nonPING_std[listindex, :len(Iext_list)], color=cmap(listindex/(len(gammaEvents_mean)-1)), alpha=.3)
        
        listindex +=1
        
ax.set_ylabel('p(Sustained Oscillations)', fontsize=7.75)

ax.set_xlabel(r'$I_{ext}$ ($\mu$ A)')
# ax.set_xticklabels([])

ax.spines[['right', 'top']].set_visible(False)
ax.legend(title=r'$E_{\text{GABA}}$ (mV), $SD_V$ (mV)', loc='right', bbox_to_anchor=(1.8, .5),ncol=1)

# ============================================================
# Add significance annotations above each Iext value
# ============================================================

for iext in Iext_plot:

    annotation = consistent_annotations.get(iext, "")

    if annotation == "":
        continue

    ax.text(
        iext,
        1.02,
        annotation,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

filenamestr='PaperFigures/Figure4E/SyncPercentZoom.png'
plt.savefig(filenamestr, dpi=600, format='png', bbox_inches='tight')





#%% Figure 4E Top

# Read desired data
E_GABA_list=[75, 73]
# E_GABA_list=[75]
NoiseAmp_list=[0, 2000]
Iext_list=[25, 27.5, 30, 32.5, 35]
gammaEvents_mean=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]
gammaEvents_std=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]
nonPING_mean=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]
nonPING_std=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]

listindex=0
for E_GABA in E_GABA_list:
    for NoiseAmp in NoiseAmp_list:
        csvstr='FinalFigure4Quants/GammaEvents_NonPINGDuration_EGABA%d_NoiseAmp%d.csv' %(E_GABA, NoiseAmp)
        temp=np.genfromtxt(csvstr, delimiter=',')
        gammaEvents_mean[listindex]=temp[:len(Iext_list),1].tolist()
        gammaEvents_std[listindex]=temp[:len(Iext_list),2].tolist()
        nonPING_mean[listindex]=temp[:len(Iext_list),3].tolist()
        nonPING_std[listindex]=temp[:len(Iext_list),4].tolist()
        listindex +=1

gammaEvents_mean=np.array(gammaEvents_mean)/18
gammaEvents_std=np.array(gammaEvents_std)/18
nonPING_mean=np.array(nonPING_mean)/18
nonPING_std=np.array(nonPING_std)

gammaEventsfreq_mean=(gammaEvents_mean)/np.array(nonPING_mean)
gammaEventsfreq_std=(gammaEvents_std)/nonPING_std

NoiseAmp_to_VoltageSD=dict([(0,0),(1,0.023),(20,0.108), (70, 0.199), (200, 0.325), (600, 0.581), (1000, 0.750), (2000,1.087), (4000,1.439), (6000, 1.842), (8000, 2.234)])


# New Plotting Ideas
cmap=cm.get_cmap('jet')

plt.rcParams['font.size']=8
fig, ax=plt.subplots()
fig.set_size_inches(2.5,2)

listindex=0
for E_GABA in E_GABA_list:
    for NoiseAmp in NoiseAmp_list:
        str1='-%d, %1.3f' % (E_GABA, NoiseAmp_to_VoltageSD[NoiseAmp])
        ax.plot(Iext_plot, gammaEventsfreq_mean[listindex, :len(Iext_list)], color=cmap(listindex/(len(gammaEvents_mean)-1)), label=str1, lw=2)
        # ax.fill_between(Iext_list, nonPING_mean[listindex, :len(Iext_list)]+nonPING_std[listindex, :len(Iext_list)],
        #                 nonPING_mean[listindex, :len(Iext_list)]-nonPING_std[listindex, :len(Iext_list)], color=cmap(listindex/(len(gammaEvents_mean)-1)), alpha=.3)
        
        listindex +=1
        
ax.set_ylabel('Gamma event rate (/s)')

ax.set_xlabel(r'$I_{ext}$ ($\mu$ A)')
# ax.set_xticklabels([])

ax.spines[['right', 'top']].set_visible(False)
ax.legend(title=r'$E_{\text{GABA}}$ (mV), $SD_V$ (mV)', loc='right', bbox_to_anchor=(1.8, .5),ncol=1)

ax.text(
    1.27, 0.1,
    '*   ' + r'$E_{\text{GABA}}$' + '\n'
    "#  Noise\n"
    "\u2020  Interaction",
    transform=ax.transAxes,
    fontsize=7,
    va="top",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="black",
        linewidth=0.5
    )
)
# ============================================================
# Add significance annotations above each Iext value
# ============================================================

for iext in Iext_plot:

    annotation = gamma_annotations.get(iext, "")

    if annotation == "":
        continue

    ax.text(
        iext,
        1.02,                    # 2% above top of axes
        annotation,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold"
    )

filenamestr='PaperFigures/Figure4/GammaRateNormalized.png'
plt.savefig(filenamestr, dpi=600, format='png', bbox_inches='tight')

