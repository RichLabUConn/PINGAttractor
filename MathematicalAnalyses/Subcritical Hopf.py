# Subcritical Hopf Module

import numpy as np

def subcritical_hopf(mu, omega, b, r_init, theta_init, dt, T):
    """
    Simulates the subcritical Hopf bifurcation system in polar coordinates.

    Parameters:
    - mu: Control parameter governing stability.
    - omega: Natural frequency term.
    - b: Nonlinear phase term.
    - r_init: Initial radius.
    - theta_init: Initial phase.
    - dt: Time step.
    - T: Total simulation time.

    Returns:
    - r, theta: Time evolution of radius and phase.
    """
    steps = int(T / dt)
    r = np.zeros(steps)
    theta = np.zeros(steps)

    # Initialize values
    r[0] = r_init
    theta[0] = theta_init

    for k in range(steps - 1):
        # Subcritical Hopf Bifurcation Equations
        dr = (mu * r[k] + r[k]**3 - r[k]**5) * dt
        dtheta = (omega + b * r[k]**2) * dt

        # Update values
        r[k + 1] = r[k] + dr
        theta[k + 1] = theta[k] + dtheta

        # Prevent overflow by clipping large values
        if np.abs(r[k+1]) > 10:
            r[k+1] = np.sign(r[k+1]) * 10  # Clip to ±10 to prevent instability

    return r, theta

def subcritical_hopf_with_noise(mu, omega, b, r_init, theta_init, dt, T, D, tau, geo_r=0.0, geo_theta=0.0):
   
    steps = int(T / dt)
    r = np.zeros(steps)
    theta = np.zeros(steps)

    # initializing r and theta
    r[0] = r_init
    theta[0] = theta_init

    # scaling factor for OU
    Ae = np.sqrt((D * tau / 2) * (1 - np.exp(-2 * dt / tau)))

    # OU noise arrays
    ou_r = np.zeros(steps)
    ou_theta = np.zeros(steps)

    for k in range(1, steps):
        # generating the noise
        dW_r = np.random.randn()
#        dW_theta = np.random.randn()

        # updating OU noise arrays
        ou_r[k] = geo_r + (ou_r[k - 1] - geo_r) * np.exp(-dt / tau) + Ae * dW_r
#        ou_theta[k] = geo_theta + (ou_theta[k - 1] - geo_theta) * np.exp(-dt / tau) + Ae * dW_theta

        # updating d_r and d_theta with OU noise
        dr = ((mu * r[k - 1] + r[k - 1]**3 - r[k - 1]**5) + ou_r[k]) * dt
        dtheta = (omega + b * r[k - 1]**2) * dt

        r[k] = r[k - 1] + dr
        theta[k] = theta[k - 1] + dtheta

        # Prevent overflow by bounding radius
        if np.abs(r[k]) > 10:
            r[k] = np.sign(r[k]) * 10

    return r, theta

import sympy as sp

def find_fixed_points(mu):
    r = sp.symbols('r', real=True, positive=True)

    # Define the equation
    eq = mu * r + r**3 - r**5

    # Solve for r
    roots = sp.solve(eq, r)

    # Convert to floats and sort
    roots = [float(root.evalf()) for root in roots]
    roots = sorted(roots)

    return roots