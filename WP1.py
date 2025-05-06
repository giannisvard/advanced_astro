from astro_constants import astro_constants as const
from scipy.optimize import newton

# Introduce dimensionless parameters

mass = const.m_sun + const.m_earth

length = const.AU

mass_parameter = const.m_earth / (const.m_sun + const.m_earth)

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
print('Classical L2 x-coordinate:', (x_L2 + mass_parameter) * length, 'm')

# https://www.sciencedirect.com/science/article/pii/S0094576524001516?via%3Dihub