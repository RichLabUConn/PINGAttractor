import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# --------------------------------------------------
# USER PARAMETERS
# --------------------------------------------------

filepaths = [
    'SpikingData_OUNewInitial_NoiseAmp0_Iext30_71.csv',
    'SpikingData_OUNewInitial_NoiseAmp2000_Iext30_57.csv',
     'SpikingData_OU_Esyni-75_NoiseAmp0_Default.csv' \
]

time_windows = [
    (6062, 6630),
    (18433, 19046),
    (4500, 5000)
]

numcells_e = 800
numcells_i = 200

T = 20000           # ms
dt = 0.02           # ms
scale = 1

fs = 1000 / (dt * scale)

nperseg = 256 * 25   # 6400
noverlap = nperseg // 2

time_column = 1
cell_column = 0

# --------------------------------------------------
# GAUSSIAN CONVOLUTION 
# --------------------------------------------------

def conv_gaussian(signal_in, srate, gauss_width):
    N_gauss = round(6 * (gauss_width / 1000) * srate)

    if (N_gauss % 2) == 0:
        N_gauss = N_gauss - 1

    x = np.linspace(-3, 3, N_gauss)
    g = np.exp(-x ** 2)

    newsig = np.convolve(signal_in, g, 'same')

    return newsig



def convert_spiketimes(spike_times, duration, srate):
    """
    Convert a list of spike times (ms) into a binary time series.

    Parameters:
    - spike_times: list of spike times in ms
    - duration: total duration in seconds
    - srate: sampling rate in Hz

    Returns:
    - times (ms), signal (0/1 array)
    """

    times = 1000 * np.arange(1/srate, duration + 1/srate, 1/srate)
    signal = np.zeros(int(duration * srate))
    Npoints = len(signal)

    for i_s in range(len(spike_times)):
        # Map spike time (ms) → index
        idx = int(np.floor(spike_times[i_s] / (duration * 1000) * Npoints))

        # Boundary protection (fix)
        if idx >= Npoints:
            idx = Npoints - 1
        elif idx < 0:
            idx = 0

        signal[idx] = 1

    return times, signal


# --------------------------------------------------
# BUILD SPIKE LISTS
# --------------------------------------------------

def build_spike_lists(spiking_csv):
    spikes_e = [[] for _ in range(numcells_e)]
    spikes_i = [[] for _ in range(numcells_i)]

    for i in range(len(spiking_csv)):
        cell = int(spiking_csv[i][cell_column])
        t = spiking_csv[i][time_column]

        if cell < numcells_e:
            spikes_e[cell].append(t)
        else:
            spikes_i[cell - numcells_e].append(t)

    return spikes_e, spikes_i

# --------------------------------------------------
# FIRING RATE (GAUSSIAN SMOOTHED)
# --------------------------------------------------

def firingRateGaussian(spikes, sd, scale):
    gaussian_sum = np.zeros(int(T / dt / scale))

    for i in range(len(spikes)):
        if i % 100 == 0:
            print(f"  Processing neuron {i}/{len(spikes)}")

        timeseries = convert_spiketimes(spikes[i], T / 1000, 1000 / dt / scale)
        gaussian = conv_gaussian(timeseries[1], 1000 / dt / scale, sd)
        gaussian_sum += gaussian

    gaussian_sum = gaussian_sum / len(spikes)

    return gaussian_sum
# --------------------------------------------------
# SPECTRUM COMPUTATION
# --------------------------------------------------

def compute_avg_spectrum(rate, window):
    t_min, t_max = window

    time = np.arange(len(rate)) * dt
    mask = (time >= t_min) & (time <= t_max)

    rate_segment = rate[mask]

    freqs, times, Sxx = signal.spectrogram(
        rate_segment,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        # scaling='density'
    )

    # Sxx = np.log10(Sxx + 1e-12)
    # mean_power = np.mean(Sxx, axis=1)
    mean_power=np.mean(Sxx,axis=1)

    return freqs, mean_power

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

spectra = []
firing_rates = []


for i in range(3):
    print(f"\nProcessing simulation {i+1}/3")

    spiking_csv = pd.read_csv(filepaths[i], header=None).values

    spikes_e, spikes_i = build_spike_lists(spiking_csv)

    firing_rate = firingRateGaussian(spikes_e, sd=2, scale=scale)

    firing_rates.append(firing_rate)   

    freqs, power = compute_avg_spectrum(firing_rate, time_windows[i])

    spectra.append((freqs, power))
    
# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
#%%

cm_to_in = 1 / 2.54

fig, axes = plt.subplots(
    1, 2,
    figsize=(100 * cm_to_in, 28 * cm_to_in)
)

ax_raw = axes[0]
ax_norm = axes[1]

labels = ["Figure 4B", "Figure 4C", "Figure 1A"]

for i, (freqs, power) in enumerate(spectra):

    mask = (freqs >= 0) & (freqs <= 150)

    freqs_plot = freqs[mask]
    power_plot = power[mask]

    # -----------------------------
    # Left panel: non-normalized
    # -----------------------------
    ax_raw.plot(
        freqs_plot,
        power_plot,
        label=labels[i],
        linewidth=8
    )

    # -----------------------------
    # Right panel: normalized
    # -----------------------------
    power_norm = power_plot / np.max(power_plot)

    ax_norm.plot(
        freqs_plot,
        power_norm,
        label=labels[i],
        linewidth=8
    )

# -----------------------------
# Left panel formatting
# -----------------------------
ax_raw.set_xlabel("Frequency (Hz)", fontsize=50)
ax_raw.set_ylabel("Power", fontsize=50)
# ax_raw.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# ax_raw.yaxis.get_offset_text().set_fontsize(32)
ax_raw.set_xlim(0, 100)
ax_raw.set_yscale('log')
ax_raw.tick_params(axis='both', labelsize=40)
# ax_raw.legend(fontsize=50)


ax_raw.spines['top'].set_visible(False)
ax_raw.spines['right'].set_visible(False)

ax_norm.spines['top'].set_visible(False)
ax_norm.spines['right'].set_visible(False)


# -----------------------------
# Right panel formatting
# -----------------------------
ax_norm.set_xlabel("Frequency (Hz)", fontsize=50)
ax_norm.set_ylabel("Normalized Power", fontsize=50)
ax_norm.set_xlim(0, 100)
ax_norm.set_ylim(0, 1.05)
ax_norm.tick_params(axis='both', labelsize=40)
ax_norm.legend(fontsize=50)


ax_raw.spines['top'].set_visible(False)
ax_raw.spines['right'].set_visible(False)

ax_norm.spines['top'].set_visible(False)
ax_norm.spines['right'].set_visible(False)




plt.tight_layout()

plt.savefig("combined_power_spectra_V3.png", dpi=600)
plt.savefig("combined_power_spectra_V3.svg")

plt.show()