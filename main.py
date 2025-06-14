from helper_functions import astro_constants as const
from scipy.optimize import newton, root, root_scalar
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Introduce dimensionless parameters

mass = const.m_sun + const.m_earth

length = const.AU

mu_se = const.m_earth / (const.m_sun + const.m_earth)

mu_em = const.m_moon / (const.m_moon + const.m_earth)

G = 6.67430e-11                             # https://ssd.jpl.nasa.gov/astro_par.html

mean_motion = np.sqrt(G * mass / length**3)

time = 1 / mean_motion

r_dust_dimensional = const.d_dust / 2

V_dust_dimensional = 4 / 3 * np.pi * r_dust_dimensional ** 3

m_dust_dimensional = const.density_dust * V_dust_dimensional # Dimensional mass of the particle

A_dust_dimensional = np.pi * r_dust_dimensional ** 2

refl = const.refl_dust

Ls = const.L_sun

c = const.c

rho_i0 = 0.3798  # Individial initial reflectivity

M_Ryugu = 4.4820e11        # kg
G  = 6.6743e-11       # m^3 kg^-1 s^-2
mu_sun = G * const.m_sun
mu_Ryugu = G * M_Ryugu
AU = const.AU   # m

r_ryugu_m = 475  # meters

LU = const.AU * (mu_Ryugu / mu_sun)**(1/3)
r_ryugu_nd = r_ryugu_m/LU

# Functions

def r1(r, mu):
    x = r[0]
    y = r[1]
    z = r[2]
    r_1 = np.array([x + mu, y, z])
    return r_1

def r2(r, mu):
    x = r[0]
    y = r[1]
    z = r[2]
    r_2 = np.array([x - (1 - mu), y, z])
    return r_2

def potential(mu, r):
    x = r[0]
    y = r[1]
    r_1 = r1(r, mu)
    r_2 = r2(r, mu)
    r1_mag = np.linalg.norm(r_1)
    r2_mag = np.linalg.norm(r_2)
    U = - ((1 - mu) / r1_mag + mu / r2_mag) - (x ** 2 + y ** 2) / 2
    return U

def dUdx(mu, r):
    x = r[0]
    r_1 = r1(r, mu)
    r_2 = r2(r, mu)
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)
    Ux1 = (1 - mu) * (x + mu) / r_1_mag ** 3
    Ux2 = mu * (x - (1 - mu)) / r_2_mag ** 3 - x
    Ux = Ux1 + Ux2
    return Ux

def dUdxx(mu, r):
    x = r[0]
    # r_1 = r1(r, mu)
    # r_2 = r2(r, mu)
    # r_1_mag = np.linalg.norm(r_1)
    # r_2_mag = np.linalg.norm(r_2)
    # Uxx1 = (1 - mu) / r_1_mag ** 3 - 3 * (1 - mu) * (x + mu) ** 2 / r_1_mag ** 5
    # Uxx2 = mu / r_2_mag ** 3 - 3 * mu * (x - (1 - mu)) ** 2 / r_2_mag ** 5
    # Uxx = Uxx1 + Uxx2 - 1
    Uxx = -(1+2*(1-mu)/(x+mu)**3 + 2*mu/(x-1+mu)**3)
    return Uxx

def dUdxy(mu, r):
    x = r[0]
    y = r[1]
    r_1 = r1(r, mu)
    r_2 = r2(r, mu)
    r_1_mag = np.linalg.norm(r_1)
    r_2_mag = np.linalg.norm(r_2)
    Uxy = -3 * (1 - mu) * (x + mu) * y / r_1_mag ** 5 - 3 * mu * (x - (1 - mu)) * y / r_2_mag ** 5
    return Uxy

def dUdyy(mu, r):
    x = r[0]
    y = r[1]
    # r_1 = r1(r, mu)
    # r_2 = r2(r, mu)
    # r_1_mag = np.linalg.norm(r_1)
    # r_2_mag = np.linalg.norm(r_2)
    # Uyy1 = (1 - mu) / r_1_mag ** 3 - mu / r_2_mag ** 3
    # Uyy2 = - 3 * (1 - mu) * y ** 2 / r_1_mag ** 5 - 3 * mu * y ** 2 / r_2_mag ** 5
    # Uyy = Uyy1 + Uyy2 - 1
    Uyy = -(1 - (1-mu)/(x+mu)**3 - mu/(x-1+mu)**3)
    return Uyy

def dUdxx_hill(r):
    x = r[0]
    r_mag = np.linalg.norm(r)
    Uxx = 3 * x ** 2 * r_mag ** -5 - r_mag ** -3 + 3
    return Uxx

def dUdyy_hill(r):
    y = r[1]
    r_mag = np.linalg.norm(r)
    Uyy = 3 * y ** 2 * r_mag ** -5 - r_mag ** -3
    return Uyy

def dUdxy_hill(r):
    x = r[0]
    y = r[1]
    r_mag = np.linalg.norm(r)
    Uxy = 3 * x * y * r_mag ** -5
    return Uxy

def acc_srp(mu, r, rho, A_dimensional, m_dimensional):
    r_1 = r1(r, mu)
    r_1_mag = np.linalg.norm(r_1)
    r_1_unit = r_1 / r_1_mag
    r_1_mag_dimensional = r_1_mag * length
    a_srp_dimensional = ((1 + rho) * Ls / (4 * np.pi * c * r_1_mag_dimensional ** 2) *
                         A_dimensional / m_dimensional * r_1_unit)
    a_srp = a_srp_dimensional * time ** 2 / length
    return a_srp

def acc_srp_beta(mu, r, beta):
    r_1 = r1(r, mu)
    r_1_mag = np.linalg.norm(r_1)
    r_1_unit = r_1 / r_1_mag
    r_1_mag_dimensional = r_1_mag * length
    a_srp = beta * (1 - mu) / r_1_mag ** 2 * r_1_unit
    return a_srp

def jacobian_classical(Uxx, Uxy, Uyy):
    A = np.zeros((4,4))
    A[0, 2] = 1
    A[1, 3] = 1
    A[2, 0] = - Uxx
    A[2, 1] = - Uxy
    A[2, 3] = 2
    A[3, 0] = - Uxy
    A[3, 1] = - Uyy
    A[3, 2] = -2
    lamda, zeta = np.linalg.eig(A)
    return A, lamda, zeta

def jacobian_srp(Uxx, Uxy, Uyy, a_srp_xx, a_srp_yy):
    A_classic = jacobian_classical(Uxx, Uxy, Uyy)[0]
    A_srp_additional = np.zeros((4,4))
    A_srp_additional[2, 0] = a_srp_xx
    A_srp_additional[3, 1] = a_srp_yy
    A = A_classic + A_srp_additional
    lamda, zeta = np.linalg.eig(A)
    return A, lamda, zeta

def jacobian_hill(state):
    xi, eta, zeta, _, _, _ = state

    r2 = xi ** 2 + eta ** 2 + zeta ** 2
    r3 = r2 ** (3 / 2)
    r5 = r2 ** (5 / 2)

    # Precompute repeated terms
    if r5 == 0:
        raise ZeroDivisionError("Radius r is zero; cannot compute Jacobian.")

    A = np.zeros((6, 6))

    # Top-right identity for velocity terms
    A[0, 3] = 1
    A[1, 4] = 1
    A[2, 5] = 1

    # ∂(ddxi)/∂(...)
    A[3, 0] = 3 - (1 / r**3 - 3 * xi ** 2 / r**5)
    A[3, 1] = (3 * xi * eta) / r**5
    A[3, 2] = (3 * xi * zeta) / r**5
    A[3, 4] = 2

    # ∂(ddeta)/∂(...)
    A[4, 0] = (3 * eta * xi) / r**5
    A[4, 1] = - (1 / r**3 - 3 * eta ** 2 / r**5)
    A[4, 2] = (3 * eta * zeta) / r**5
    A[4, 3] = -2

    # ∂(ddzeta)/∂(...)
    A[5, 0] = (3 * zeta * xi) / r**5
    A[5, 1] = (3 * zeta * eta) / r**5
    A[5, 2] = - (1 / r**3 - 3 * zeta ** 2 / r**5 + 1)

    return A


def a_srp_xx(mu, r_0, rho, A_dimensional, m_dimensional):
    x_0 = r_0[0]
    r_1 = r1(r_0, mu)
    r_1_mag = np.linalg.norm(r_1)
    C_dimensional = (1 + rho) * Ls / (4 * np.pi * c) * A_dimensional / m_dimensional
    C = C_dimensional * time ** 2 / length ** 3
    a_xx = C * r_1_mag ** - 3 - 3 * C * (x_0 + mu) ** 2 * r_1_mag ** - 5
    return a_xx

def a_srp_yy(mu, r_0, rho, A_dimensional, m_dimensional):
    r_1 = r1(r_0, mu)
    r_1_mag = np.linalg.norm(r_1)
    C_dimensional = (1 + rho) * Ls / (4 * np.pi * c) * A_dimensional / m_dimensional
    C = C_dimensional * time ** 2 / length ** 3
    a_yy = C * r_1_mag ** - 3
    return a_yy

def beta_to_rho(beta, mu_sun, m, A):
    rho = 4 * np.pi * c * beta * mu_sun * m / (Ls * A) - 1
    return rho

def kappa(rho):
    k = 15.6044 * (1 + rho)  # nondimensional SRP acceleration
    return k

def eom_hill(t, state):
    k = kappa(rho_i0)
    xi, eta, zeta, xidot, etadot, zetadot = state
    r = np.linalg.norm(state[0:3])
    ddxi =  2*etadot + 3*xi   - xi   / r**3 + k
    ddeta = -2*xidot        - eta  / r**3
    ddzeta =      - zeta    - zeta / r**3
    return np.array([xidot, etadot, zetadot, ddxi, ddeta, ddzeta])

# def differential_corrections(Phi, state_half):
#     vel = state_half[3:6]  # ẋ, ẏ, ż
#     acc = eom_hill(0, state_half)[3:6]  # ẍ, ÿ, z̈
#     x_dot_half, y_dot_half, z_dot_half = vel
#     x_dot_dot_half, y_dot_dot_half, z_dot_dot_half = acc
#     c1 = Phi[3,2] - x_dot_dot_half / y_dot_half * Phi[1,2]
#     c2 = Phi[3,4] - x_dot_dot_half / y_dot_half * Phi[1,4]
#     c3 = Phi[5,2] - z_dot_dot_half / y_dot_half * Phi[1,2]
#     c4 = Phi[5,4] - z_dot_dot_half / y_dot_half * Phi[1,4]
#     dy_dot_0 = (c4 - c2 * c3 / c1) ** -1 * (c3 / c1 * x_dot_half - z_dot_half)
#     dz_0 = - x_dot_half / c1 - c2 / c1 * (c4 - c2 * c3 / c1) ** -1 * (c3 / c1 * x_dot_half - z_dot_half)
#     dt_half = - 1 / y_dot_half * (Phi[1,2] * dz_0 + Phi[1,4] * dy_dot_0)
#     return dy_dot_0, dz_0, dt_half

def differential_corrections(Phi, state_half):
    vel = state_half[3:6]  # ẋ, ẏ, ż
    acc = eom_hill(0, state_half)[3:6]  # ẍ, ÿ, z̈
    x_dot_half, y_dot_half, z_dot_half = vel
    x_dot_dot_half, y_dot_dot_half, z_dot_dot_half = acc
    error_vector = np.array([-x_dot_half, -z_dot_half])
    M1 = np.zeros((2, 2))
    M1[0,0] = Phi[3,2]
    M1[0,1] = Phi[3,4]
    M1[1,0] = Phi[5,2]
    M1[1,1] = Phi[5,4]
    M2 = np.zeros((2, 2))
    M2[0, 0] = Phi[0, 2] * x_dot_dot_half
    M2[0, 1] = Phi[0, 4] * x_dot_dot_half
    M2[1, 0] = Phi[0, 2] * z_dot_dot_half
    M2[1, 1] = Phi[0, 4] * z_dot_dot_half
    M2 = - 1 / x_dot_half * M2
    M = M1 + M2
    delta = np.linalg.solve(M, error_vector)
    dy_dot_0, dz_0 = delta
    return dy_dot_0, dz_0

def eom_hill_corrector(t, state):
    state_derivative = eom_hill(t, state[0:6])
    A = jacobian_hill(state[0:6]) # Jacobian, must be numpy wrt hill with srp
    Phi = np.reshape(state[6:], (6, 6))
    Phi_dot = A @ Phi
    Phi_dot = Phi_dot.reshape(-1)
    state_dot = np.append(state_derivative, Phi_dot)
    return state_dot

def differential_corrector(state_0, T_guess):
    iteration = 1
    error = 1
    state_history_corrected = None
    max_iterations = 20
    while error >= 1e-10 and iteration <= max_iterations:
        solution = solve_ivp(eom_hill_corrector, (0, T_guess), state_0, rtol=1e-12, atol=1e-12, dense_output=True)
        state_mid = solution.sol(T_guess / 2)
        Phi = np.reshape(state_mid[6:], (6, 6))
        dy_dot_0, dz_0 = differential_corrections(Phi, state_mid[0:6])
        state_0[2] += 0.9 * dz_0
        state_0[4] += 0.9 * dy_dot_0
        error_x_dot = state_mid[3]
        error_z_dot = state_mid[5]
        error = np.sqrt(error_x_dot ** 2 + error_z_dot ** 2)
        state_history_corrected = solution.y[0:6]
        print('Iteration', iteration, 'error', error)
        iteration += 1
    return state_history_corrected





#########################################################
# WP1 ###################################################
#########################################################

# Classical L2

def f_mu_Earth_Sun(x):
    r = np.array([x, 0, 0])
    f = dUdx(mu_se, r)
    return f

def f_prime_mu_Earth_Sun(x):
    r = np.array([x, 0, 0])
    f = dUdxx(mu_se, r)
    return f

x_L2_0 = 1.01 # Initial guess for L2, a bit further in the x-axis than Earth

x_L2 = newton(f_mu_Earth_Sun, x_L2_0, f_prime_mu_Earth_Sun, disp=True)

r_L2 = np.array([x_L2, 0, 0])

print('Classical L2 x-coordinate:', x_L2, 'AU')
# print('Classical L2 x-coordinate wrt Earth:', x_L2 - (1 - mu_se), 'AU')
# print('Classical L2 x-coordinate wrt Sun:', x_L2 + mu_se, 'AU')

# Verification (L1)

def f_mu_Earth_Moon(x):
    r = np.array([x, 0, 0])
    f = dUdx(mu_em, r)
    return f

def f_prime_mu_Earth_Moon(x):
    r = np.array([x, 0, 0])
    f = dUdxx(mu_em, r)
    return f

x_L2_0_em = 1.1

x_L2_em = newton(f_mu_Earth_Moon, x_L2_0_em, f_prime_mu_Earth_Moon, disp=True)

r_L2_em = np.array([x_L2_em, 0, 0])

print('Classical L2 x-coordinate from Earth-Moon system:', x_L2_em, 'AU')

# Sub L2

def f_mu_Earth_Sun_srp_vector(r):
    a_srp = acc_srp(mu_se, r, refl, A_dust_dimensional, m_dust_dimensional)
    f = np.array([dUdx(mu_se, r) - a_srp[0], 0, 0])
    return f

# x_L2_srp = root_scalar(f_mu_Earth_Sun_srp, method='secant', x0=1.09, x1=1.12)

r_L2_0 = np.array([1.01, 0, 0]) # Initial guess for L2, a bit further in the x-axis than Earth

r_L2_srp = root(f_mu_Earth_Sun_srp_vector, r_L2_0, tol=1e-10).x

x_L2_srp = r_L2_srp[0]

print('Sub-L2 x-coordinate:', x_L2_srp, 'AU')
# print('Sub-L2 x-coordinate wrt Earth:', x_L2_srp - (1 - mu_se), 'AU')
# print('Sub-L2 x-coordinate wrt Sun:', x_L2_srp + mu_se, 'AU')

# Verification (sub-L1)

beta = 0.0363

def f_mu_Earth_Sun_srp_vector_verification(r):
    a_srp = acc_srp_beta(mu_se, r, beta)
    f = np.array([dUdx(mu_se, r) - a_srp[0], 0, 0])
    return f

r_L1_0 = np.array([0.99, 0, 0]) # Initial guess for L1, a bit further in the x-axis than Earth

r_L1_srp = root(f_mu_Earth_Sun_srp_vector_verification, r_L1_0, tol=1e-10).x

x_L1_srp = r_L1_srp[0]

print('Sub-L1 x-coordinate:', x_L1_srp, 'AU')

diameter_increments = list(np.arange(1, 11) * 1e-2)

def f_mu_Earth_Sun_srp_vector_diameter(r, D):
    r_dust_dimensional = D / 2
    A_dust_dimensional = np.pi * r_dust_dimensional ** 2
    V_dust_dimensional = 4 / 3 * np.pi * r_dust_dimensional ** 3
    m_dust_dimensional = V_dust_dimensional * const.density_dust
    a_srp = acc_srp(mu_se, r, refl, A_dust_dimensional, m_dust_dimensional)
    # print('x:', r[0], 'AU')
    # print('acc srp:', a_srp)
    # print('dUdx:', dUdx(mu_se, r))
    f = np.array([dUdx(mu_se, r) - a_srp[0], 0, 0])
    return f

x_L2_srp_array = []

for D in diameter_increments:
    wrapped_func_srp_L2 = lambda r: f_mu_Earth_Sun_srp_vector_diameter(r, D)
    r_L2_increments = root(wrapped_func_srp_L2, r_L2_0, tol=1e-10).x
    x_L2_srp_array.append(r_L2_increments[0])

diameter_increments_centi = np.array(diameter_increments) * 1e2

# Plot
plt.figure(figsize=(8, 5))
plt.plot(diameter_increments_centi, x_L2_srp_array, marker='o')
plt.xlabel('Dust Particle Diameter (cm)', fontsize=14)
plt.ylabel('x-coordinate of L2 (-)', fontsize=16)
plt.title('Sensitivity Analysis on the Location of L2 \n Varying Particle Diameter', fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
ax = plt.gca()
ax.xaxis.get_offset_text().set_fontsize(14)
ax.yaxis.get_offset_text().set_fontsize(14)
plt.tight_layout()
plt.savefig('figures/sensitivity_analysis_L2.png', dpi=300, bbox_inches='tight')
plt.close()

#########################################################
# WP2 ###################################################
#########################################################



A_classic, lamda_classic, zeta_classic = jacobian_classical(dUdxx(mu_se, r_L2), dUdxy(mu_se, r_L2), dUdyy(mu_se, r_L2))

print('Classical L2 eigenvalues:', lamda_classic)

A_srp, lamda_srp, zeta_srp = jacobian_srp(
    dUdxx(mu_se, r_L2_srp), dUdxy(mu_se, r_L2_srp), dUdyy(mu_se, r_L2_srp),
    a_srp_xx(mu_se, r_L2_srp, refl, A_dust_dimensional, m_dust_dimensional),
    a_srp_yy(mu_se, r_L2_srp, refl, A_dust_dimensional, m_dust_dimensional)
)

print('Sub-L2 eigenvalues:', lamda_srp)

#########################################################
# WP3 ###################################################
#########################################################

# Dust properties
density    = 1200   # kg/m^3
Cr0   = 0.5
Cr_personal  = 0.4333
R_particle = r_dust_dimensional
A_area    = np.pi * (R_particle)**2
m_mass    = density * (4/3) * np.pi * (R_particle)**3
P0 = 1e17


# Compute kappa values
k1 = (1 + Cr0) * P0 / (mu_sun ** (2 / 3) * mu_Ryugu ** (1 / 3)) * (A_area / m_mass)
k2 = (1 + Cr_personal) * P0 / (mu_sun ** (2 / 3) * mu_Ryugu ** (1 / 3)) * (A_area / m_mass)

def f_ref(r):
    return 3 * r ** 3 + k1 * r ** 2 - 1

def df_ref(r):
    return 9 * r ** 2 + 2 * k1 * r

def f_pers(r):
    return 3 * r ** 3 + k2 * r ** 2 - 1

def df_pers(r):
    return 9 * r ** 2 + 2 * k2 * r

# Find the three xi values
xi_cl = 0.6933612744
xi_ref = newton(f_ref, 1.0, df_ref, disp=True)
xi_perso = newton(f_pers, 1.0, df_pers, disp=True)

xi_vals = [xi_cl, xi_ref, xi_perso]
labels = ['classical', 'Reflection 0.5', 'Reflection 0.4333 ']
print("Reflection 0.5: ", xi_ref)
print(f"Reflection {Cr_personal}: ", xi_perso)
print("Reflection 0.5: ", xi_ref*LU, 'm')
print(f"Reflection {Cr_personal}: ", xi_perso*LU, 'm')


# Exercise 3.2

# For each xi, build Jacobian A and compute eigenvalues
results = {}
for label, xi in zip(labels, xi_vals):
    eta = 0.0
    r = np.sqrt(xi**2 + eta**2)
    Uxx = 3 - 1/r**3 + 3*xi**2/r**5
    Uyy = -1/r**3 + 3*eta**2/r**5
    Uxy = 3*xi*eta/r**5

    # Hill‐problem linearization matrix
    A_mat = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [Uxx, Uxy, 0, 2],
        [Uxy, Uyy, -2, 0]
    ])

    eigvals, _ = np.linalg.eig(A_mat)
    results[label] = eigvals

# Print neatly
print(f"{'Label':15s}  {'eig1':>14s}  {'eig2':>14s}  {'eig3':>14s}  {'eig4':>14s}")
print("-" * 75)
for label in labels:
    ev = results[label]
    formatted = "  ".join(f"{np.real(λ):+7.4f}{np.imag(λ):+7.4f}i" for λ in ev)
    print(f"{label:15s}  {formatted}")

#########################################################
# WP4 ###################################################
#########################################################

# Uncorrected orbit

# Inidividual initial conditions
xi0, eta0, zeta0 = 0.196745558941, 0, 0.074201147580
xidot0, etadot0, zetadot0 = 0, -0.789119271940, 0
state0 = np.array([xi0, eta0, zeta0, xidot0, etadot0, zetadot0])

# Integrate over one synodic period
T_guess = 0.59
sol = solve_ivp(eom_hill, (0, T_guess), state0, rtol=1e-12, atol=1e-12)
xi, eta, zeta = sol.y[0], sol.y[1], sol.y[2]

r_ryugu_hill = r_ryugu_nd


# Font sizes
fontsize_title = 18
fontsize_labels = 18
fontsize_ticks = 16
fontsize_legend = 16

# Helper to make 3D axes equal
def set_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    for i, set_lim in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        set_lim([origin[i] - radius, origin[i] + radius])

# Helper to draw Ryugu as a 3D sphere
def plot_ryugu_sphere(ax, radius):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='red', alpha=0.4)

# General limit calculator
def get_limit(a, b):
    return 1.1 * np.max(np.abs(np.concatenate([a, b])))

# --- Figure 1: 3D View ---
fig1 = plt.figure(figsize=(7, 7))
ax3d = fig1.add_subplot(111, projection='3d')
ax3d.plot(xi, zeta, eta, label='Orbit')
plot_ryugu_sphere(ax3d, r_ryugu_hill)
ax3d.scatter(xi_perso, 0, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax3d.set_xlabel('ξ', fontsize=fontsize_labels, labelpad=10)
ax3d.set_ylabel('ζ', fontsize=fontsize_labels, labelpad=10)
ax3d.set_zlabel('η', fontsize=fontsize_labels, labelpad=10)
ax3d.set_title('3D View of Periodic Orbit', fontsize=fontsize_title, pad=20)
ax3d.tick_params(labelsize=fontsize_ticks)
ax3d.set_box_aspect([1, 1, 1])
set_equal_3d(ax3d)
ax3d.legend(fontsize=fontsize_legend)
fig1.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
plt.savefig('figures/3D_hill_scaled.png', dpi=300)
plt.close()

# --- Figure 2: (ξ, η) Projection ---
lim_x_xy = np.max(np.abs(xi)) * 1.1
lim_y_eta = np.max(np.abs(eta)) * 1.1
lim_xy = max(lim_x_xy, lim_y_eta)

fig2 = plt.figure(figsize=(7, 7))
ax_xy = fig2.add_subplot(111)
ax_xy.plot(xi, eta)
circle_xy = plt.Circle((0, 0), r_ryugu_hill, color='red', alpha=0.4, label='Ryugu')
ax_xy.add_patch(circle_xy)
ax_xy.scatter(xi_perso, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax_xy.set_xlim(0, lim_xy)
ax_xy.set_ylim(-lim_xy * 0.5, lim_xy * 0.5)
ax_xy.set_aspect('equal', 'box')
ax_xy.xaxis.set_major_locator(MultipleLocator(0.05))
ax_xy.yaxis.set_major_locator(MultipleLocator(0.05))
ax_xy.set_xlabel('ξ', fontsize=fontsize_labels)
ax_xy.set_ylabel('η', fontsize=fontsize_labels)
ax_xy.set_title('(ξ, η)-Projection', fontsize=fontsize_title)
ax_xy.tick_params(labelsize=fontsize_ticks)
ax_xy.grid(True)
ax_xy.legend(fontsize=fontsize_legend)
fig2.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('figures/xi_eta_scaled.png', dpi=300)
plt.close()

# --- Figure 3: (ξ, ζ) Projection ---
lim_x_xz = np.max(np.abs(xi)) * 1.1
lim_y_zeta = np.max(np.abs(zeta)) * 1.1
lim_xz = max(lim_x_xz, lim_y_zeta)

fig3 = plt.figure(figsize=(7, 7))
ax_xz = fig3.add_subplot(111)
ax_xz.plot(xi, zeta)
circle_xz = plt.Circle((0, 0), r_ryugu_hill, color='red', alpha=0.4, label='Ryugu')
ax_xz.add_patch(circle_xz)
ax_xz.scatter(xi_perso, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax_xz.set_xlim(0, lim_xz)
ax_xz.set_ylim(-lim_xz * 0.5, lim_xz * 0.5)
ax_xz.set_aspect('equal', 'box')
ax_xz.xaxis.set_major_locator(MultipleLocator(0.05))
ax_xz.yaxis.set_major_locator(MultipleLocator(0.05))
ax_xz.set_xlabel('ξ', fontsize=fontsize_labels)
ax_xz.set_ylabel('ζ', fontsize=fontsize_labels)
ax_xz.set_title('(ξ, ζ)-Projection', fontsize=fontsize_title)
ax_xz.tick_params(labelsize=fontsize_ticks)
ax_xz.grid(True)
ax_xz.legend(fontsize=fontsize_legend)
fig3.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('figures/xi_zeta_scaled.png', dpi=300)
plt.close()

# --- Figure 4: (η, ζ) Projection ---
lim_x_yz = np.max(np.abs(eta)) * 1.2
lim_y_yz = np.max(np.abs(zeta)) * 1.2
lim_yz = max(lim_x_yz, lim_y_yz)

fig4 = plt.figure(figsize=(7, 7))
ax_yz = fig4.add_subplot(111)
ax_yz.plot(eta, zeta)
circle_yz = plt.Circle((0, 0), r_ryugu_hill, color='red', alpha=0.4, label='Ryugu')
ax_yz.add_patch(circle_yz)
ax_yz.scatter(0, 0, color='blue', marker='o', s=40, label='Sub-L2')
ax_yz.set_xlim(-lim_yz * 1.2, lim_yz * 1.2)
ax_yz.set_ylim(-lim_yz * 1.2, lim_yz * 1.2)
ax_yz.set_aspect('equal', 'box')
ax_yz.xaxis.set_major_locator(MultipleLocator(0.05))
ax_yz.yaxis.set_major_locator(MultipleLocator(0.05))
ax_yz.set_xlabel('η', fontsize=fontsize_labels)
ax_yz.set_ylabel('ζ', fontsize=fontsize_labels)
ax_yz.set_title('(η, ζ)-Projection', fontsize=fontsize_title)
ax_yz.tick_params(labelsize=fontsize_ticks)
ax_yz.grid(True)
ax_yz.legend(fontsize=fontsize_legend)
fig4.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
plt.savefig('figures/eta_zeta_scaled.png', dpi=300)
plt.close()


# Corrected Orbit

# Phi0 = np.eye(6)
# Phi0 = Phi0.reshape(-1)
# state0 = np.append(state0, Phi0)
#
#
# state_corrected = differential_corrector(state0, T_guess)


# # --- Figure 1: 3D View ---
# fig1 = plt.figure(figsize=(6, 5))
# ax3d = fig1.add_subplot(111, projection='3d')
# ax3d.plot(Y[0], Y[1], Y[2], label='Orbit')
# ax3d.scatter(0, 0, 0, color='red', marker='s', s=50, label='Ryugu')
# ax3d.scatter(xi_perso, 0, 0, color='blue', marker='o', s=40, label='Sub-L2')
# ax3d.set_xlabel('ξ')
# ax3d.set_ylabel('ζ')
# ax3d.set_zlabel('η')
# ax3d.set_title('3D View of Periodic Orbit')
# ax3d.set_box_aspect([1, 1, 1])
# ax3d.legend()
# plt.savefig('figures/3D_hill_corrected.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# # --- Figure 2: (ξ, η) Projection ---
# fig2 = plt.figure(figsize=(6, 5))
# ax_xy = fig2.add_subplot(111)
# ax_xy.plot(Y[0], Y[1])
# ax_xy.scatter(0, 0, color='red', marker='s', s=50, label='Ryugu')
# ax_xy.scatter(xi_perso, 0, color='blue', marker='o', s=40, label='Sub-L2')
# ax_xy.set_aspect('equal', 'box')
# ax_xy.set_xlabel('ξ')
# ax_xy.set_ylabel('η')
# ax_xy.set_title('(ξ, η)-Projection')
# ax_xy.grid(True)
# ax_xy.legend()
# plt.savefig('figures/xi_eta_corrected.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# # --- Figure 3: (ξ, ζ) Projection ---
# fig3 = plt.figure(figsize=(6, 5))
# ax_xz = fig3.add_subplot(111)
# ax_xz.plot(Y[0], Y[2])
# ax_xz.scatter(0, 0, color='red', marker='s', s=50, label='Ryugu')
# ax_xz.scatter(xi_perso, 0, color='blue', marker='o', s=40, label='Sub-L2')
# ax_xz.set_aspect('equal', 'box')
# ax_xz.set_xlabel('ξ')
# ax_xz.set_ylabel('ζ')
# ax_xz.set_title('(ξ, ζ)-Projection')
# ax_xz.grid(True)
# plt.savefig('figures/xi_zeta_corrected.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# # --- Figure 4: (η, ζ) Projection ---
# fig4 = plt.figure(figsize=(6, 5))
# ax_yz = fig4.add_subplot(111)
# ax_yz.plot(Y[1], Y[2])
# ax_yz.scatter(0, 0, color='red', marker='s', s=50, label='Ryugu')
# ax_yz.scatter(0, 0, color='blue', marker='o', s=40, label='Sub-L2')
# ax_yz.set_aspect('equal', 'box')
# ax_yz.set_xlabel('η')
# ax_yz.set_ylabel('ζ')
# ax_yz.set_title('(η, ζ)-Projection')
# ax_yz.grid(True)
# ax_yz.legend()
# plt.savefig('figures/eta_zeta_corrected.png', dpi=300, bbox_inches='tight')
# plt.close()