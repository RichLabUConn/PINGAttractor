Code to simulate the network under the two distinct initial conditions studied (default and "sparsely active" initial conditions") are included in this folder.

By default, the code will run with noise amplitude D=0 for each of the E_GABA values defined by esyn_i_min:esyn_i_step:esyn_i_max. These can be changed by altering the """Input Variables""" portion of the preamble in concert with the code's input. The code will output:
1. A "StatsOverEsynI ... .csv" file compiling average network statistics for each E_GABA value.
2. For each individual run, a "SpikingData ... .csv" file containing the spike times.
3. A rough visualization of each individual simulation using baseline quantifications of network activity over moving time windows ("StatsOverTime ... .png").

"DefaultIniaitalConditions.py" includes the inital conditions used in Figures 1-3 of the manuscript.
