import os
import numpy as np

# =========================
# Define parameter ranges
# =========================
e_gaba_values = np.arange(-75, -64, 1)   # -75 to -65
I_ext_values = np.arange(25, 55, 5)      # 25 to 50

# Input directory
data_dir = "CSV_Gamma"

# Output directory
output_dir = "Output_Gamma"
os.makedirs(output_dir, exist_ok=True)

# =========================
# Preallocate result matrices
# =========================
shape = (len(e_gaba_values), len(I_ext_values))

gamma_mean = np.zeros(shape)
gamma_std = np.zeros(shape)

sync_mean = np.zeros(shape)
sync_std = np.zeros(shape)

nonPING_mean = np.zeros(shape)
nonPING_std = np.zeros(shape)

# =========================
# Main loops
# =========================
for i, e_gaba in enumerate(e_gaba_values):
    for j, I_ext in enumerate(I_ext_values):

        gammaEventRate = []
        syncProportion = []
        nonPINGduration = []

        for number in range(1, 101):

            filename = f"GammaEventRate_SyncPercent_OUNewInitial_EGaba{e_gaba}_Iext{I_ext}_{number}.csv"
            filepath = os.path.join(data_dir, filename)

            if os.path.exists(filepath):
                try:
                    array = np.genfromtxt(filepath, delimiter=',')
                    array = np.atleast_1d(array)

                    gammaEventRate.append(array[0])
                    nonPINGduration.append(array[1])
                    syncProportion.append(array[2])

                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Missing: {filename}")

        # Convert to arrays
        gammaEventRate = np.array(gammaEventRate)
        nonPINGduration = np.array(nonPINGduration)
        syncProportion = np.array(syncProportion)

        # Compute stats (NaN-safe)
        gamma_mean[i, j] = np.nanmean(gammaEventRate)
        gamma_std[i, j] = np.nanstd(gammaEventRate)

        nonPING_mean[i, j] = np.nanmean(nonPINGduration)
        nonPING_std[i, j] = np.nanstd(nonPINGduration)

        sync_mean[i, j] = np.nanmean(syncProportion)
        sync_std[i, j] = np.nanstd(syncProportion)

# =========================
# Save outputs to Output_Gamma
# =========================
np.savetxt(os.path.join(output_dir, "gamma_mean.csv"), gamma_mean, delimiter=',')
np.savetxt(os.path.join(output_dir, "gamma_std.csv"), gamma_std, delimiter=',')

np.savetxt(os.path.join(output_dir, "sync_mean.csv"), sync_mean, delimiter=',')
np.savetxt(os.path.join(output_dir, "sync_std.csv"), sync_std, delimiter=',')

np.savetxt(os.path.join(output_dir, "nonPING_mean.csv"), nonPING_mean, delimiter=',')
np.savetxt(os.path.join(output_dir, "nonPING_std.csv"), nonPING_std, delimiter=',')

# Save parameter axes
np.savetxt(os.path.join(output_dir, "e_gaba_values.csv"), e_gaba_values, delimiter=',')
np.savetxt(os.path.join(output_dir, "I_ext_values.csv"), I_ext_values, delimiter=',')

print(f"All matrices saved to {output_dir}")