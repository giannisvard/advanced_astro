from helper_functions import astro_constants as const
from scipy.optimize import newton, root
import numpy as np
import matplotlib.pyplot as plt

# Introduce dimensionless parameters

mass = const.m_sun + const.m_earth

length = const.AU

mu = const.m_earth / (const.m_sun + const.m_earth)

G = 6.67430e-11                             # https://ssd.jpl.nasa.gov/astro_par.html

mean_motion = np.sqrt(G * mass / length**3)

time = 1 / mean_motion

# Introduce numerically solved equations

def f_mu_Earth_Sun(x):
    r1 = abs(x + mu)               # distance to Sun at -mu
    r2 = abs(x - (1 - mu))         # distance to Earth at 1 - mu
    f = x - (1 - mu)/r1**3 * (x + mu) - mu/r2**3 * (x - (1 - mu))
    return f

def fprime_mu_Earth_Sun(x):
    r1 = abs(x + mu)
    r2 = abs(x - (1 - mu))
    f_term1 = 3 * (1 - mu) * (x + mu) / r1 ** 4 - (1 - mu) / r1 ** 3
    f_term2 = 3 * mu / r2 ** 4 * (x - (1 - mu)) - mu / r2 ** 3
    f = 1 + f_term1 + f_term2
    return f


x_L2_0 = 1.01 # Initial guess for L2, a bit further in the x-axis than Earth

x_L2 = newton(f_mu_Earth_Sun, x_L2_0, fprime_mu_Earth_Sun, tol=1e-20, disp=True)

print('Classical L2 x-coordinate:', x_L2, 'AU')
print('Classical L2 x-coordinate:', x_L2 * length, 'm')
print('Classical L2 x-coordinate wrt Earth:', (x_L2 - (1 - mu)) * length, 'm')
print('Classical L2 x-coordinate wrt Earth:', x_L2 - (1 - mu), 'AU')

# Wakker p. 67
# https://www.researchgate.net/publication/304609071_Earth-Moon_L1_libration_point_orbit_continuous_stationkeeping_control_using_time-varying_LQR_and_backstepping

def srp_acceleration_dust(r, P, A, m, reflectivity):    # Might have to use the beta coefficient for clarity of the verification
    a = (1 + reflectivity) * P * A / m * r / np.linalg.norm(r)
    return a

def srp_acceleration_dust_L2(r_dimensionless, D):
    x = r_dimensionless[0]
    y = r_dimensionless[1]
    z = r_dimensionless[2]
    r1_dimensionless = np.array([x + mu, y, z])
    sfu_1AU = 1367                  # https://www.sciencedirect.com/topics/engineering/solar-radiation-pressure
    A = np.pi * (D / 2) ** 2
    rho = 1.2e3
    V = 4 / 3 * np.pi * (D / 2) ** 3
    m = V * rho
    refl = 0.5
    r1_dimensional = r1_dimensionless * length
    P = sfu_1AU * const.AU ** 2 / (const.c * np.linalg.norm(r1_dimensional) ** 2)    # https://ntrs.nasa.gov/api/citations/20205005240/downloads/AAS_srpframeworkpaper_Final02.pdf#:~:text=Psrp%20%3D%20P0%20c%20%12,57%20%C3%97%2010%E2%88%926%20%12
    a_dimensional = srp_acceleration_dust(r1_dimensional, P, A, m, refl)
    a_dimensionless = a_dimensional * time ** 2 / length
    return a_dimensionless

def srp_acceleration_beta(r, beta, sun_grav_param, inc_angle, sail_vec):
    r_mag = np.linalg.norm(r)
    a = beta * sun_grav_param / r_mag ** 2 * np.cos(inc_angle) ** 2 * sail_vec
    return a

def srp_acceleration_beta_L1(r_dimensionless):
    x = r_dimensionless[0]
    y = r_dimensionless[1]
    z = r_dimensionless[2]
    r1_dimensionless = np.array([x + mu, y, z])
    r1_dimensional = r1_dimensionless * length
    beta = 0.0363
    sun_grav_param = G * const.m_sun
    inc_angle = 0
    sail_vec = np.array([1, 0, 0])
    a_dimensional = srp_acceleration_beta(r1_dimensional, beta, sun_grav_param, inc_angle, sail_vec)
    a_dimensionless = a_dimensional * time ** 2 / length
    return a_dimensionless

def potential_gradient(r):
    x = r[0]
    y = r[1]
    z = r[2]
    r1 = np.linalg.norm([x + mu, y, z])
    r2 = np.linalg.norm([x - (1 - mu), y, z])
    dUdx = x - (1 - mu) / r1 ** 3 * (x + mu) - mu / r2 ** 3 * (x - (1 - mu))
    dUdy = y * (1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3)
    dUdz = z * ((1 - mu) / r1 ** 3 + mu / r2 ** 3)
    DU = np.array([dUdx, dUdy, dUdz])
    return DU

def f_mu_Earth_Sun_srp(r, D):
    f = srp_acceleration_dust_L2(r, D) + potential_gradient(r)
    return f

def f_mu_Earth_Sun_srp_verification(r):
    f = srp_acceleration_beta_L1(r) + potential_gradient(r)
    return f

r_L2_0 = np.array([1.01, 0, 0]) # Initial guess for L2, a bit further in the x-axis than Earth

D_fixed = 10e-2

wrapped_func_srp_L2 = lambda r: f_mu_Earth_Sun_srp(r, D_fixed)

r_L2 = root(wrapped_func_srp_L2, r_L2_0, tol=1e-20).x

print('SRP L2 coordinate:', r_L2, 'AU')
print('SRP L2 x-coordinate:', r_L2 * length, 'm')
print('SRP L2 x-coordinate wrt Sun:', (r_L2 + mu) * length, 'm')

r_L1_0 = np.array([0.9, 0, 0])

r_L1 = root(f_mu_Earth_Sun_srp_verification, r_L1_0, tol=1e-20).x

print('SRP L1 coordinate (beta = 0.0363):', r_L1, 'AU')
print('SRP L1 x-coordinate (beta = 0.0363):', r_L1 * length, 'm')
print('SRP L1 x-coordinate wrt Sun (beta = 0.0363):', (r_L1 + mu) * length, 'm')

diameter_increments = list(np.arange(1, 11) * 1e-2)

r_L2_array_x = []

for D in diameter_increments:
    wrapped_func_srp_L2 = lambda r: f_mu_Earth_Sun_srp(r, D)
    r_L2 = root(wrapped_func_srp_L2, r_L2_0, tol=1e-20).x
    r_L2_array_x.append(r_L2[0])

# Plot
plt.figure(figsize=(8, 5))
plt.plot(diameter_increments, r_L2_array_x, marker='o')
plt.xlabel('D (Diameter)')
plt.ylabel('r_L2[1] (Second Element of r_L2)')
plt.title('r_L2[1] vs Diameter')
plt.grid(True)
plt.show()