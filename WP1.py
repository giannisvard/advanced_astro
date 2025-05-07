from astro_constants import astro_constants as const
from scipy.optimize import newton, root
import numpy as np

# Introduce dimensionless parameters

mass = const.m_sun + const.m_earth

length = const.AU

mass_parameter = const.m_earth / (const.m_sun + const.m_earth)

G = 6.67430e-11                             # https://ssd.jpl.nasa.gov/astro_par.html

mean_motion = np.sqrt(G * mass / length**3)

time = 1 / mean_motion

# Introduce numerically solved equations

def f_mu_Earth_Sun(x):
    mu = mass_parameter
    r1 = abs(x + mu)               # distance to Sun at -mu
    r2 = abs(x - (1 - mu))         # distance to Earth at 1 - mu
    f = x - (1 - mu)/r1**3 * (x + mu) - mu/r2**3 * (x - (1 - mu))
    return f

def fprime_mu_Earth_Sun(x):
    mu = mass_parameter
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    f_term1 = 3 * (1 - mu) * (x + mu) / r1 ** 4 - (1 - mu) / r1 ** 3
    f_term2 = 3 * mu / r2 ** 4 * (x - (1 - mu)) - mu / r2 ** 3
    f = 1 + f_term1 + f_term2
    return f


x_L2_0 = 1.01 # Initial guess for L2, a bit further in the x-axis than Earth

x_L2 = newton(f_mu_Earth_Sun, x_L2_0, fprime_mu_Earth_Sun, tol=1e-10, disp=True)

print('Classical L2 x-coordinate:', x_L2, 'AU')
print('Classical L2 x-coordinate:', x_L2 * length, 'm')

# https://www.sciencedirect.com/science/article/pii/S0094576524001516?via%3Dihub

def srp_acceleration_dust(r, P, A, m, reflectivity):
    a = (1 + reflectivity) * P * A / m * r
    return a

def srp_acceleration_dust_L2(r_dimensionless):
    P = 382.8e24                        # https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
    D = 10e-2
    A = np.pi * (D / 2) ** 2
    rho = 1.2e3
    V = 4 / 3 * np.pi * (D / 2) ** 3
    m = V * rho
    refl = 0.5
    r_dimensional = r_dimensionless * length
    a_dimensional = srp_acceleration_dust(r_dimensional, P, A, m, refl)
    a_dimensionless = a_dimensional * time ** 2 / length
    return a_dimensionless


def potential(r):


def f_mu_Earth_Sun_srp(r):
    f =
r_L2_0 = np.array([1.01, 0, 0]) # Initial guess for L2, a bit further in the x-axis than Earth

r_L2 = root(srp_acceleration_dust_L2, r_L2_0, tol=1e-10).x

print('SRP L2 coordinate:', r_L2, 'AU')
print('SRP L2 distance from the Sun:', (r_L2 + mass_parameter), 'AU')
