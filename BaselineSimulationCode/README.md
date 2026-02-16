Code to simulate the network under the two distinct initial conditions studied (default and "sparsely active" initial conditions") are included in this folder.

"DefaultInititalConditions.py" includes the inital conditions used in Figures 1-3 of the manuscript.

By default, the code will run with noise amplitude D=0 for each of the E_GABA values defined by esyn_i_min:esyn_i_step:esyn_i_max. These can be changed by altering the """Input Variables""" portion of the preamble in concert with the code's input. The code will output:
1. A "StatsOverEsynI ... .csv" file compiling average network statistics for each E_GABA value.
2. For each individual run, a "SpikingData ... .csv" file containing the spike times.
3. A rough visualization of each individual simulation using baseline quantifications of network activity over moving time windows ("StatsOverTime ... .png").

"SparselyFiringInitialConditions.py" implements the initial conditions used in Figure 4 of the manuscript. It includes additional quantifications using the firing rate histogram. Rather than loop over a range of E_GABA values, this code by default sets E_GABA=-75 mV and instead loops over a range of mean external input values (Iext_e_mean). Thus, instead of a "StatsOverEsynI ... .csv" file this code outputs a "Stats_OUNewInitial ... .csv" file compiling average network statistics for each input.
