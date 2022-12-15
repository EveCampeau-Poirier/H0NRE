# %%
import h5py
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c

c = c.to('Mpc/d')  # Speed of light

import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior
from pypolychord.priors import GaussianPrior

import getdist.plots as plots

import torch
from networks import SetTransformer
from func import r_estimator, normalization, get_time_delays


# ----------------------------------- FUNCTION --------------------------------------
# NORM
def norm(x, y):
    return np.sqrt(x ** 2 + y ** 2)


# ROTATION MATRIX
def mtx_rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])


def log_gaussian(x, mu, sigma):
    D = x.shape[0]
    log_gauss = -.5 * np.sum((x - mu) ** 2) / sigma ** 2 - D / 2 * np.log(2 * np.pi * sigma ** 2)
    return log_gauss


# CARTESIAN COORDINATES TO SPHERICAL
def cart2pol(x, y):
    return norm(x, y), np.arctan2(y, x)


# SPHERICAL COORDINATES TO CARTESIAN
def pol2cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def get_Fermat_potentials(x0_lens, y0_lens, f, phi, theta_E, gamma_ext, phi_ext, xim_fov, yim_fov, x0_AGN, y0_AGN):
    f_prime = np.sqrt(1 - f ** 2)

    # image coordinates in the lens coordinate system
    xim, yim = np.tensordot(mtx_rot(-phi), np.array([xim_fov - x0_lens, yim_fov - y0_lens]), axes=1)
    r_im, phi_im = cart2pol(yim, xim)
    # AGN coordinates in the lens coordinate system
    xsrc, ysrc = np.tensordot(mtx_rot(-phi), np.array([x0_AGN - x0_lens, y0_AGN - y0_lens]), axes=1)
    # Coordinate translation for the external shear
    xtrans, ytrans = np.tensordot(mtx_rot(-phi), np.array([x0_lens, y0_lens]), axes=1)

    # External shear in the lens coordinate system
    gamma1_fov, gamma2_fov = pol2cart(gamma_ext, phi_ext)
    gamma1, gamma2 = np.tensordot(mtx_rot(-2 * phi), np.array([gamma1_fov, gamma2_fov]), axes=1)

    # X deflection angle (eq. 27a Kormann et al. 1994)
    def alphax(varphi):
        return theta_E * np.sqrt(f) / f_prime * np.arcsin(f_prime * np.sin(varphi))

    # Y deflection angle (eq. 27a Kormann et al. 1994)
    def alphay(varphi):
        return theta_E * np.sqrt(f) / f_prime * np.arcsinh(f_prime / f * np.cos(varphi))

    # Lens potential deviation from a SIS (Meneghetti, 19_2018, psi_tilde)
    def psi_tilde(varphi):
        return np.sin(varphi) * alphax(varphi) + np.cos(varphi) * alphay(varphi)

    # Lens potential (eq. 26a Kormann et al. 1994)
    def lens_pot(varphi):
        return psi_tilde(varphi) * radial(varphi)

    # Shear potential (eq. 3.80 Meneghetti)
    def shear_pot(x, y):
        return gamma1 / 2 * (x ** 2 - y ** 2) + gamma2 * x * y

    # Radial componant (eq. 34 Kormann et al. 1994)
    def radial(varphi):
        usual_rad = ysrc * np.cos(varphi) + xsrc * np.sin(varphi) + psi_tilde(varphi)
        translation = gamma1 * (xtrans * np.sin(varphi) - ytrans * np.cos(varphi)) + gamma2 * (
                xtrans * np.cos(varphi) + ytrans * np.sin(varphi))
        shear_term = 1 + gamma1 * (np.cos(varphi) ** 2 - np.sin(varphi) ** 2) - 2 * gamma2 * np.cos(
            varphi) * np.sin(varphi)
        return (usual_rad + translation) / shear_term

    # Fermat potential
    geo_term = norm(xim - xsrc, yim - ysrc) ** 2 / 2
    lens_term = lens_pot(phi_im)
    shear_term = shear_pot(xim + xtrans, yim + ytrans)
    fermat_pot = geo_term - lens_term - shear_term

    fermat_pot = fermat_pot - np.min(fermat_pot)

    return fermat_pot


def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


# -------------------------------- GLOBAL VARIABLES ---------------------------------

path_out = "results/PolyChord/new_frame/"

# import keys
keys = h5py.File("data_NRE/keys.hdf5", 'r')
test_keys = keys["test"][:]
keys.close()

# import data
dataset = h5py.File("data_NRE/dataset.hdf5", 'r')
H0 = dataset["Hubble_cst"][test_keys]
lower_bound = np.floor(np.min(H0))
upper_bound = np.ceil(np.max(H0))

# remove out of bounds data
selected = np.array([20, 42, 33, 34, 43, 36, 13, 32, 10, 5])
idx_up = np.where(H0 > 75.)[0]
idx_down = np.where(H0 < 65.)[0]
idx_out = np.concatenate((idx_up, idx_down))
H0 = np.delete(H0, idx_out, axis=0)
H0 = H0[selected].flatten()
test_keys = np.delete(test_keys, idx_out, axis=0)
true_dt = dataset["time_delays"][test_keys][selected]
true_pot = dataset["Fermat_potential"][test_keys][selected]
z = dataset["redshifts"][test_keys][selected]
im_pos = dataset["positions"][test_keys][selected]
param = dataset["parameters"][test_keys][selected, :7]
dataset.close()

sig_dt = .35
sig = 5e-4

nsamp = H0.shape[0]
npts = 7500
nDerived = 0


# ----------------------------------- NRE --------------------------------------

model = torch.load("results/SetTransformer/prior_H0/bignet/models/trained_model.pt")

dt = true_dt[true_dt != 0].reshape(nsamp, 3)
pot = true_pot[true_pot != 0].reshape(nsamp, 3)
samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)

# observations
samples_rep = torch.repeat_interleave(torch.from_numpy(samples), npts, dim=0)
z_rep = torch.repeat_interleave(torch.from_numpy(z), npts, dim=0)

# NRE posterior
grid = np.linspace(lower_bound + 1, upper_bound - 1, npts).reshape(npts, 1)
grid_tile = np.tile(grid, (nsamp, 1))
grid = grid.flatten()

dataset_test = torch.utils.data.TensorDataset(samples_rep, torch.from_numpy(grid_tile), z_rep)
dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=200)
post_NRE = []
for samp, hbl, shift in dataloader:
    ratios = r_estimator(model, samp, hbl, shift)
    post_NRE.extend(ratios)

post_NRE = np.asarray(post_NRE).reshape(nsamp, npts)
grid_tile = grid_tile.reshape(nsamp, npts)
post_NRE = normalization(grid_tile, post_NRE)


# ----------------------------------- Nested Sampling --------------------------------------

for i in range(nsamp):

    xim = im_pos[i, 0][true_dt[i] != -1.]
    yim = im_pos[i, 1][true_dt[i] != -1.]
    nim = xim.shape[0]
    dt_obs = true_dt[i][true_dt[i] > 0.]

    nDims = int(param[i].shape[0] + 2 * nim + 3) # 2*nim : xim, yim ; 3 : x0_AGN, y0_AGn, H0
    settings = PolyChordSettings(nDims, nDerived)
    settings.nlive = 500
    settings.do_clustering = True
    settings.read_resume = False

    settings.file_root = 'new_frame{}'.format(i)

    mu = np.concatenate((param[i], xim, yim, np.zeros(2)))

    def prior(hypercube):
        x = hypercube.copy()
        x[0] = UniformPrior(lower_bound + 1, upper_bound - 1)(x[0])
        x[1:] = GaussianPrior(mu, sig)(x[1:])
        return x

    def likelihood(theta):
        cosmo_model = FlatLambdaCDM(H0=theta[0], Om0=.3)
        Ds = cosmo_model.angular_diameter_distance(z[i, 1])
        Dd = cosmo_model.angular_diameter_distance(z[i, 0])
        Dds = cosmo_model.angular_diameter_distance_z1z2(z[i, 0], z[i, 1])

        pot = get_Fermat_potentials(theta[1], theta[2], theta[3], theta[4],
                                    theta[5], theta[6], theta[7],  # param
                                    theta[8: 8 + nim],  # xim
                                    theta[8 + nim: 8 + 2 * nim],  # yim
                                    theta[8 + 2 * nim],  # xsrc
                                    theta[9 + 2 * nim])  # ysrc
        pot = pot[pot != 0].flatten()
        dt = get_time_delays(pot, z[i, 0], Dd, Ds, Dds)

        I = log_gaussian(dt_obs, dt, sig_dt)

        return I, []

    truth = np.concatenate((np.array([H0[i]]), mu), axis=0)
    lkh = likelihood(truth)
    print(i, lkh)

    start = time.time()
    output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)
    print(f'Integration completed {((time.time() - start) // 60):.0f}m {((time.time() - start) % 60):.0f}s')

    paramnames = [("H0", r"H_{0}"),
                  ('xlens', r"x_{lens}"), ("ylens", r"y_{lens}"), ("f", r"f"), ("phi", r"\phi"),
                  ("theta_E", r"\theta_E"), ("gammaext", r"\gamma_{ext}"), ("phiext", r"\phi_{ext}")]
    for j in range(nim):
        paramnames += [(f"xim{j}", r"x_{im"+f"{j}"+r"}")]
    for j in range(nim):
        paramnames += [(f"yim{j}", r"y_{im"+f"{j}"+r"}")]
    paramnames += [("xsrc", r"x_{src}"), ("ysrc", r"y_{src}")]
    output.make_paramnames_files(paramnames)

    marks = {paramnames[j][0]: truth[j] for j in range(nDims)}

    high = truth + 5 * sig
    low = truth - 5 * sig
    limits = {"H0": [lower_bound + 1, upper_bound - 1]}
    limits = {paramnames[j][0]: [low[j], high[j]] for j in range(1, nDims)}

    post_NS = output.posterior
    g = plots.getSubplotPlotter()
    g.triangle_plot(post_NS, filled=True, markers=marks, title_limit=1, param_limits=limits)
    g.export(path_out + 'corner_plot{}.pdf'.format(i))


    # ----------------------------------- Figure --------------------------------------

    plt.style.use(['science', 'vibrant'])
    g = plots.get_subplot_plotter(width_inch=3, subplot_size_ratio=.776)
    g.settings.linewidth = 2
    g.plot_1d(post_NS, 'H0', lims=[lower_bound+1, upper_bound-1], normalized=True, colors=['C1'], ls=['--'])
    g.add_x_marker(H0[i], ls='dotted', lw=1.5)
    plt.plot(grid, post_NRE[i], '-', color='C4', zorder=0, linewidth=2)
    if i == 5:
        g.add_legend(['Nested' + '\n' + 'sampling', 'Truth', 'NRE'])
    if i in [0, 5]:
        plt.ylabel(r"$p(H_0 \mid \Delta t, \Delta \phi)$")
    else:
        plt.ylabel("")
    plt.xlabel("")
    g.export(path_out + 'post{}.pdf'.format(i))
