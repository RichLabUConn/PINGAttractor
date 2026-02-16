# Wilson Cowan Module

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm

def default_pars(**kwargs):
    pars = {}

    # Excitatory parameters
    pars['tau_E'] = 20.0     # Timescale of the E population [ms]
    pars['a_E'] = 1.0      # Gain of the E population
    pars['theta_E'] = 5.0  # Threshold of the E population

    # Inhibitory parameters
    pars['tau_I'] = 10.0    # Timescale of the I population [ms]
    pars['a_I'] = 1.0      # Gain of the I population
    pars['theta_I'] = 20.0  # Threshold of the I population

    # Connection strength
    pars['wEE'] = 16.0   # E to E
    pars['wEI'] = 26.0   # I to E
    pars['wIE'] = 20.0  # E to I
    pars['wII'] = 1.0  # I to I

    # External input
    pars['I_ext_E'] = 2.0
    pars['I_ext_I'] = 7.0

    # Simulation parameters
    pars['T'] = 1000        # Total duration of simulation [ms]
    pars['dt'] = 0.1        # Simulation time step [ms]
    pars['rE_init'] = 0.2  # Initial value of E
    pars['rI_init'] = 0.2  # Initial value of I

    # External parameters if any
    for k in kwargs:
        pars[k] = kwargs[k]

    # Vector of discretized time points [ms]
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

    return pars

def F(x, a, theta):
    # Clip the inputs to avoid overflow
    max_exp = 700  # np.exp(700) is approximately the largest value before overflow
    exp_input_neg = np.clip(-a * (x - theta), -max_exp, max_exp)
    exp_input_pos = np.clip(a * theta, -max_exp, max_exp)

    # Calculate the response function with clipped values
    f = 1 / (1 + np.exp(exp_input_neg)) - 1 / (1 + np.exp(exp_input_pos))
    return f

def dF(x, a, theta):
    dFdx = a * np.exp(-a * (x - theta)) * (1 + np.exp(-a * (x - theta)))**-2
    return dFdx

def F_inv(x, a, theta):
    # Calculate Finverse (ln(x) can be calculated as np.log(x))
    F_inverse = -1/a * np.log((x + (1 + np.exp(a * theta))**-1)**-1 - 1) + theta
    return F_inverse

def EIderivs(rE, rI, tau_E, a_E, theta_E, wEE, wEI, I_ext_E, tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    # Compute the derivative of rE
    drEdt = (-rE + F(wEE * rE - wEI * rI + I_ext_E, a_E, theta_E)) / tau_E
    # Compute the derivative of rI
    drIdt = (-rI + F(wIE * rE - wII * rI + I_ext_I, a_I, theta_I)) / tau_I
    return drEdt, drIdt

def simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wEI, wIE, wII, I_ext_E, I_ext_I, rE_init, rI_init, dt, range_t, **other_pars):
    # Initialize activity arrays
    Lt = range_t.size
    rE = np.append(rE_init, np.zeros(Lt - 1))
    rI = np.append(rI_init, np.zeros(Lt - 1))
    I_ext_E = I_ext_E * np.ones(Lt)
    I_ext_I = I_ext_I * np.ones(Lt)

    # Simulate the Wilson-Cowan equations
    for k in range(Lt - 1):
        # Calculate the derivative of the E population
        drE = dt / tau_E * (-rE[k] + F(wEE * rE[k] - wEI * rI[k] + I_ext_E[k], a_E, theta_E))
        # Calculate the derivative of the I population
        drI = dt / tau_I * (-rI[k] + F(wIE * rE[k] - wII * rI[k] + I_ext_I[k], a_I, theta_I))
        # Update using Euler's method
        rE[k + 1] = rE[k] + drE
        rI[k + 1] = rI[k] + drI

    return rE, rI

def my_fp(pars, rE_init, rI_init):
    """
    Use opt.root function to solve Equations (2)-(3) from initial values
    """

    tau_E, a_E, theta_E = pars['tau_E'], pars['a_E'], pars['theta_E']
    tau_I, a_I, theta_I = pars['tau_I'], pars['a_I'], pars['theta_I']
    wEE, wEI = pars['wEE'], pars['wEI']
    wIE, wII = pars['wIE'], pars['wII']
    I_ext_E, I_ext_I = pars['I_ext_E'], pars['I_ext_I']

    # Define the right hand of Wilson-Cowan equations
    def my_WCr(x):
        rE, rI = x
        drEdt = (-rE + F(wEE * rE - wEI * rI + I_ext_E, a_E, theta_E)) / tau_E
        drIdt = (-rI + F(wIE * rE - wII * rI + I_ext_I, a_I, theta_I)) / tau_I
        y = np.array([drEdt, drIdt])
        return y

    x0 = np.array([rE_init, rI_init])
    x_fp = opt.root(my_WCr, x0).x

    return x_fp

def check_fp(pars, x_fp, mytol=1e-6):
    """
    Verify (drE/dt)^2 + (drI/dt)^2 < mytol

    Args:
        pars    : Parameter dictionary
        x_fp    : value of fixed point
        mytol   : tolerance, default as 10^{-6}

    Returns :
        Whether it is a correct fixed point: True/False
    """

    drEdt, drIdt = EIderivs(x_fp[0], x_fp[1], **pars)
    return drEdt**2 + drIdt**2 < mytol

def my_plot_trajectories(pars, start, stop, dx):
    """
    Plots trajectories in the phase plane with custom start, stop, and dx values.
    
    Parameters:
    - pars: Dictionary of Wilson-Cowan parameters.
    - start: The starting value for initial conditions.
    - stop: The stopping value for initial conditions.
    - dx: Step size for selecting initial conditions.
    """
    pars = pars.copy()
    
    # Generate rE and rI initial condition grid based on start, stop, and dx
    rE_init_vals = np.arange(start, stop, dx)
    rI_init_vals = np.arange(start, stop, dx)
    
    # Define known fixed points
    x_fp_1 = my_fp(pars, 0.1, 0.8)  # Stable oscillatory FP (GREEN)
    x_fp_2 = my_fp(pars, 0.5, 0.9)  # Dashed line boundary FP (RED)
    x_fp_3 = my_fp(pars, 0.8, 0.9)  # High activity stable FP (BLUE)
    
    for rE_init in rE_init_vals:
        for rI_init in rI_init_vals:
            pars['rE_init'], pars['rI_init'] = rE_init, rI_init
            rE_tj, rI_tj = simulate_wc(**pars)
            
            # Determine trajectory color based on which fixed point it converges to
            final_point = (rE_tj[-1], rI_tj[-1])
            
            if np.linalg.norm(np.array(final_point) - np.array(x_fp_1)) < 0.05:  # Close to FP1
                color = '#9467bd'
            elif np.linalg.norm(np.array(final_point) - np.array(x_fp_2)) < 0.05:  # Close to FP2
                color = '#ff7f0e'
            elif np.linalg.norm(np.array(final_point) - np.array(x_fp_3)) < 0.05:  # Close to FP3
                color = '#1f77b4'
            else:
                color = 'gray'  # Fallback for unclassified trajectories

            plt.plot(rE_tj, rI_tj, color, alpha=0.8)

    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')

def plot_fp(x_fp, fp_index, position=(-0.09, 0.05), rotation=0, color='black'):
    """
    Plot a fixed point with a unique label and custom color.

    Args:
        x_fp (tuple): Fixed point coordinates (rE, rI).
        fp_index (int): Index of the fixed point (1, 2, 3, ...).
        position (tuple): Offset for text placement.
        rotation (int): Rotation angle for text.
        color (str): Color of the fixed point marker and label.
    """
    # Plot the fixed point with a user-specified color
    plt.plot(x_fp[0], x_fp[1], 'o', ms=8, color=color, label=f'Fixed Point {fp_index}')
