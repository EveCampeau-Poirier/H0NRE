# -*- coding: utf-8 -*-
"""
Lensing simulation with a SIE lens and gaussian host galaxy and AGN
Ève Campeau-Poirier, may the fourth, 2021
"""
# ------------------------------ Librairies ----------------------------------

# Numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
# Scipy
from scipy.special import ellipkinc
from scipy.signal import correlate2d
from scipy.interpolate import griddata
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter
from scipy.ndimage import interpolation
# Astropy
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from astropy import units as u


# --------------------------------- FUNCTIONS ----------------------------------

# NORM
def norm(x, y):
    return np.sqrt(x ** 2 + y ** 2)


# ROTATION MATRIX
def mtx_rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi), np.cos(phi)]])


# GAUSSIAN
def gauss(x, y, A, sigma, mux, muy):
    return A * np.exp(-((x - mux) ** 2 + (y - muy) ** 2) / 2 / sigma ** 2) / 2 / np.pi / sigma ** 2


# CARTESIAN COORDINATES TO SPHERICAL
def cart2pol(x, y):
    return norm(x, y), np.arctan2(y, x)


# SPHERICAL COORDINATES TO CARTESIAN
def pol2cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# MAGNITUDE TO AMPLITUDE
def magn2amp(magn):
    return 10 ** (-(magn - 25.9463) / 2.5)


c = c.to('Mpc/d')  # Speed of light


##############################################################################

class training_set(object):
    def __init__(self,
                 nsamp=55000,  # Number of examples to simulate
                 npix=96,  # Number of pixel on the image side
                 dim=7.68,  # Image dimension (arcsec)
                 sig_psf=0.15,  # Gaussian PSF extent
                 lens_density="SIE",  # Lens profil
                 source_luminosity="GAUSSIAN",  # Host light profil
                 analytique=True,  # Analytical alpha angles or convolution
                 seed=2,  # random seed
                 path=""
                 ):

        self.nsamp = nsamp
        self.npix = npix
        self.dim = dim
        self.dimpix = dim / npix
        self.sig_psf = sig_psf
        self.lens_density = lens_density
        self.source_luminosity = source_luminosity
        self.analytique = analytique
        self.seed = seed
        self.path = path

        np.random.seed(self.seed)

    # --------------------------- Lens angular position ---------------------------

    def lens_angular_position(self, param):
        """Coordinates centered and aligned with SIE axes"""

        x0_lens, y0_lens, phi = param[0], param[1], param[3]

        xlens = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix - x0_lens
        ylens = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix - y0_lens
        lens_ang = np.asarray(np.meshgrid(xlens, ylens))
        lens_ang = np.tensordot(mtx_rot(-phi), lens_ang, axes=1)
        norm_lens_ang = norm(lens_ang[0], lens_ang[1])

        return lens_ang, norm_lens_ang

    # ------------------------------ kappa map ------------------------------------

    def SIE_kappa_map(self, param):
        """SIE surface mass density according to eq (21a), R. Kormann et al."""

        ellip, theta_E = param[2], param[4]

        # Compute min axis
        lens_ang, norm_lens_ang = self.lens_angular_position(param)
        min_axis = norm(ellip * lens_ang[0], lens_ang[1])
        # Selection of the diverging pixel in the center
        inf = np.where(norm_lens_ang == 0.)
        min_axis[inf] = 1.

        # Compute kappa map
        kappa_map = np.sqrt(ellip) * theta_E / 2 / min_axis
        # Mean of diverging pixel
        moy = theta_E / 2 * np.sqrt(ellip) * self.dimpix / 2 * ellipkinc(2 * np.pi, 1 - ellip ** 2)
        kappa_map[inf] = moy / np.pi / (self.dimpix / 2) ** 2

        return kappa_map

    def fig_kappa(self, kappa_map, param, test):
        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]

        plt.figure()
        plt.imshow(np.log10(kappa_map), extent=[-self.dim / 2, self.dim / 2,
                                                -self.dim / 2, self.dim / 2],
                   origin='lower', cmap='gist_ncar')
        plt.title(r"Lens : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_lens, y0_lens) +
                  r"f={:.1f}, $\theta$$_E$={:.1f}, ".format(ellip, theta_E) +
                  r"$\phi$={:.2f}$\pi$".format(phi / np.pi))
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")
        cbar = plt.colorbar()
        cbar.set_label(r'$\log\kappa(\vec{\theta})$', rotation=270,
                       fontsize=10, labelpad=15)
        plt.clim(np.min(np.log10(kappa_map)), np.max(np.log10(kappa_map)))

        if test:
            self.plot_cuts_and_caustics(param)

    # ---------------------------- Analytical alphas -----------------------------

    def SIE_alpha(self, param):
        """SIE alpha angles according to eq (27a), R. Kormann et al."""
        # Parameters
        ellip, phi, theta_E = param[2:5]
        f = ellip
        f_prime = np.sqrt(1 - f ** 2)

        # Polar coordinates centered and aligned with SIE axes
        lens_ang, norm_lens_ang = self.lens_angular_position(param)
        indx, indy = np.where(norm_lens_ang == 0.)
        norm_lens_ang[indx, indy] = 1.
        sin_phi = lens_ang[0] / norm_lens_ang
        cos_phi = lens_ang[1] / norm_lens_ang
        sin_phi[indx, indy] = 0.
        cos_phi[indx, indy] = 0.

        # Deflection angles (alpha)
        alphax_th = theta_E * np.sqrt(f) / f_prime * np.arcsin(f_prime * sin_phi)
        alphay_th = theta_E * np.sqrt(f) / f_prime * np.arcsinh(f_prime / f * cos_phi)
        alpha_th = np.array([alphax_th, alphay_th])

        # Restoring coordinate system
        alpha_th = np.tensordot(mtx_rot(phi), alpha_th, axes=1)

        return alpha_th

    # ------------------------------ Alphas convolution ----------------------------

    def alpha_convolution(self, kappa_map):
        """Calcul de la déviation alpha selon eq (3.8) de Bartelmann et Schneider"""

        ### Ajout d'une ligne et d'une colonne pour la symétrie
        # pad=np.zeros((self.npix+1,self.npix+1)) # Initialisation du pad
        # pad[:-1,:-1]=kappa_map # Ajout de la kappa_map
        # pad[-1,:-1]=kappa_map[0,:] # Ajout de la colonne
        # pad[:-1,-1]=kappa_map[:,0] # Ajout de la ligne
        # kappa_map=pad # Remplacement

        x_range = np.linspace(-1., 1., self.npix * 2 + 1) * self.dim  # Intervalle des x (arcsec)
        y_range = np.linspace(-1., 1., self.npix * 2 + 1) * self.dim  # Intervalle des y (arcsec)
        vec = np.asarray(np.meshgrid(x_range, y_range))  # Ensemble des vecteurs

        denom = vec[0] ** 2 + vec[1] ** 2  # Dénominateur des intégrants
        zero = np.where(denom == 0)  # Sélection de sa valeur 0 pour éviter inf
        denom[zero] = 1.  # Remplacement du zéro
        integrant_x = -vec[0] / denom  # Calcul de l'intégrant en x
        integrant_y = -vec[1] / denom  # Calcul de l'intégrant en y
        integrant_x[zero] = 0  # Remplacement de la fausse valeur à la divergence
        integrant_y[zero] = 0  # Remplacement de la fausse valeur à la divergence

        # Calcul de la composante x d'alpha par convolution
        alphax = correlate2d(kappa_map, integrant_x, mode='same') * self.dim ** 2 / (self.npix - 1) ** 2 / np.pi
        # alphax=alphax[:-1,:-1] # Retrait de la ligne et de la colonne ajoutées

        # Calcul de la composante y d'alpha par convolution
        alphay = correlate2d(kappa_map, integrant_y, mode='same') * self.dim ** 2 / (self.npix - 1) ** 2 / np.pi
        # alphay=alphay[:-1,:-1] # Retrait de la ligne et de la colonne ajoutées

        alpha = np.array([alphax, alphay])

        return alpha

    def fig_alpha(self, alpha, param):

        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]
        alphax, alphay = alpha
        alpha_norm = norm(alphax, alphay)
        xlens = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        ylens = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        coor = np.asarray(np.meshgrid(xlens, ylens))

        plt.figure()
        plt.imshow(alpha_norm, extent=[-self.dim / 2, self.dim / 2,
                                       -self.dim / 2, self.dim / 2],
                   origin='lower', cmap='gist_ncar')
        cbar = plt.colorbar()
        cbar.set_label(r'$||\vec{\alpha}||$ (arcsec)', rotation=270,
                       fontsize=10, labelpad=15)
        plt.clim(np.min(alpha_norm), np.max(alpha_norm))
        plt.quiver(coor[0], coor[1], alphax, alphay)
        plt.title(r"Lens : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_lens, y0_lens) +
                  r"f={:.1f}, $\theta$$_E$={:.1f}, ".format(ellip, theta_E) +
                  r"$\phi$={:.2f}$\pi$".format(phi / np.pi))
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")

    # ------------------------------ External shear --------------------------------

    def external_shear(self, param):
        """Deflection angles caused by the external shear"""

        # Parameters
        gamma_ext, phi_ext = param[5:7]
        gamma1 = gamma_ext * np.cos(phi_ext)
        gamma2 = gamma_ext * np.sin(phi_ext)
        # Shear matrix
        ext_shear = np.array([[gamma1, gamma2],
                              [gamma2, -gamma1]])

        # Image plane coordinates
        xcoor = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        ycoor = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        coor = np.asarray(np.meshgrid(xcoor, ycoor))

        return np.tensordot(ext_shear, coor, axes=1)

    # ---------------------------------- Source -----------------------------------

    def gaussian_host(self, param):
        """Gaussian light profil"""

        # Parameters
        m_host, sig_host, x0_host, y0_host = param[7:11]
        A_host = magn2amp(m_host)

        # Source plane coordinates
        xsrc = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        ysrc = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        src_ang = np.asarray(np.meshgrid(xsrc, ysrc))

        # Count per second
        light_profil = gauss(src_ang[0], src_ang[1], A_host, sig_host, x0_host, y0_host)
        host = light_profil * self.dimpix ** 2

        return host

    def fig_host(self, host, param):

        m_host, sig_host, x0_host, y0_host = param[7:11]

        plt.figure()
        plt.imshow(host, extent=[-self.dim / 2, self.dim / 2, -self.dim / 2, self.dim / 2],
                   origin='lower', cmap='hot')
        plt.title(r"Host : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_host, y0_host) +
                  r"m={:.1f}, $\sigma$={:.2f}".format(m_host, sig_host))
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")
        cbar = plt.colorbar()
        cbar.set_label(r'CPS', rotation=270, fontsize=10,
                       labelpad=15)
        plt.clim(np.min(host), np.max(host))

    # ---------------------------------- Image ------------------------------------

    def host_image(self, alpha, host):
        """Host galaxy image"""

        # Image plane coordinates
        xtheta = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        ytheta = np.arange(-self.npix / 2, self.npix / 2, 1) * self.dimpix
        theta = np.asarray(np.meshgrid(xtheta, ytheta))

        # Lens equation
        beta = theta - alpha

        # Interpolation
        values = host.flatten()
        pts_x = theta[0].flatten()
        pts_y = theta[1].flatten()
        points = np.column_stack((pts_x, pts_y))
        grid_x = beta[0].flatten()
        grid_y = beta[1].flatten()
        grid = np.column_stack((grid_x, grid_y))
        interpolation = griddata(points, values, grid, method='linear')
        im = interpolation.reshape(self.npix, self.npix)

        return beta, im

    def fig_image(self, im, param):

        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]
        m_host, sig_host, x0_host, y0_host = param[7:11]

        plt.figure()
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")
        plt.imshow(im, extent=[-self.dim / 2, self.dim / 2, -self.dim / 2, self.dim / 2],
                   origin='lower', cmap='hot')  # , norm=LogNorm()
        plt.title(r"Lens : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_lens, y0_lens) +
                  r"f={:.1f}, $\theta$$_E$={:.1f}, ".format(ellip, theta_E) +
                  r"$\phi$={:.2f}$\pi$".format(phi / np.pi) + "\n" +
                  r"Host : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_host, y0_host) +
                  r"m={:.1f}, $\sigma$={:.2f}".format(m_host, sig_host))
        cbar = plt.colorbar()
        cbar.set_label(r'CPS', rotation=270, fontsize=10,
                       labelpad=15)
        plt.clim(np.min(im), np.max(im))

    # ---------------------------- Solving lens equation ---------------------------

    def AGN_images(self, param):
        """Computes the position and the magnification of AGN images
        Also outputs the Fermat potential at these positions"""

        # Parameters
        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]
        gamma_ext, phi_ext = param[5:7]
        x0_AGN, y0_AGN = param[-2:]
        f = ellip
        f_prime = np.sqrt(1 - f ** 2)

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

        # Equations to solve (Eq. 35 Kormann et al. 1994)
        def y_lens_eq(varphi):
            x = radial(varphi)
            return ysrc - x * np.cos(varphi) + alphay(varphi) - gamma1 * (x * np.cos(varphi) + ytrans) + gamma2 * (
                        x * np.sin(varphi) + xtrans)

        def x_lens_eq(varphi):
            x = radial(varphi)
            return xsrc - x * np.sin(varphi) + alphax(varphi) + gamma1 * (x * np.sin(varphi) + xtrans) + gamma2 * (
                        x * np.cos(varphi) + ytrans)

        def roots_func(varphi):
            return [y_lens_eq(varphi), x_lens_eq(varphi)]

        # Root function evaluated on its whole domain
        varphi = np.linspace(0, 2 * np.pi, 1000)
        yval = y_lens_eq(varphi)
        xval = x_lens_eq(varphi)
        ysign = np.sign(yval)
        xsign = np.sign(xval)

        # Search for roots
        r_im = []
        phi_im = []
        for i in range(len(varphi) - 1):
            if ysign[i] != ysign[i + 1] and xsign[i] != xsign[i + 1]:
                phi_root = brentq(y_lens_eq, varphi[i], varphi[i + 1])
                val_root = y_lens_eq(phi_root)
                if np.isnan(val_root) or abs(val_root) > 1e-3:
                    continue
                r_root = radial(phi_root)
                if r_root > 0:
                    r_im = np.append(r_im, r_root)
                    phi_im = np.append(phi_im, phi_root)
            elif ysign[i] != ysign[i + 1] and (xval[i] + xval[i + 1]) / 2 < 1e-4:
                phi_root = brentq(y_lens_eq, varphi[i], varphi[i + 1])
                val_root = x_lens_eq(phi_root)
                if np.isnan(val_root) or abs(val_root) > 1e-3:
                    continue
                r_root = radial(phi_root)
                if r_root > 0:
                    r_im = np.append(r_im, r_root)
                    phi_im = np.append(phi_im, phi_root)
            elif (yval[i] + yval[i + 1]) / 2 < 1e-4 and xsign[i] != xsign[i + 1]:
                phi_root = brentq(x_lens_eq, varphi[i], varphi[i + 1])
                val_root = y_lens_eq(phi_root)
                if np.isnan(val_root) or abs(val_root) > 1e-3:
                    continue
                r_root = radial(phi_root)
                if r_root > 0:
                    r_im = np.append(r_im, r_root)
                    phi_im = np.append(phi_im, phi_root)

        if len(phi_im) > 1:
            if abs(phi_im[-1] - phi_im[0] - 2 * np.pi) < 1e-1 or abs(phi_im[-1] - phi_im[-2]) < 1e-1:
                phi_im = phi_im[:-1]
                r_im = r_im[:-1]
            if np.any(np.diff(phi_im) < 1e-1):
                ind = np.where(np.diff(phi_im) < 1e-1)
                phi_im = np.delete(phi_im, ind)
                r_im = np.delete(r_im, ind)

        # Magnification
        cos = np.cos(phi_im)
        sin = np.sin(phi_im)
        kappa = np.sqrt(ellip) * theta_E / 2 / r_im / norm(cos, ellip * sin)
        A = np.array([[1 - 2 * kappa * sin ** 2 + gamma1, kappa * np.sin(2 * phi_im) - gamma2],
                      [kappa * np.sin(2 * phi_im) - gamma2, 1 - 2 * kappa * cos ** 2 - gamma1]], dtype='float')
        det = np.linalg.det(np.moveaxis(A, 2, 0))
        magnification = 1 / det

        # Cartesian coordinates (lens coordinate system)
        yim, xim = pol2cart(r_im, phi_im)

        # Fermat potential
        geo_term = norm(xim - xsrc, yim - ysrc) ** 2 / 2
        lens_term = lens_pot(phi_im)
        shear_term = shear_pot(xim + xtrans, yim + ytrans)
        fermat_pot = geo_term - lens_term - shear_term

        # Transformation into the image plane coordinate system
        im_pos_lens = np.array([xim, yim])
        lens_pos_fov = np.array([x0_lens, y0_lens])
        xim_fov, yim_fov = np.tensordot(mtx_rot(phi), im_pos_lens, axes=1) + lens_pos_fov.reshape(2, 1)

        return xim_fov, yim_fov, magnification, fermat_pot

    # ------------------------------------ PSF -------------------------------------

    def gaussian_PSF(self, image, sigma):
        """Applies a gaussian PSF"""
        psf_im = gaussian_filter(image, sigma / self.dimpix, mode='nearest', truncate=2.)
        return psf_im

    # ----------------------------- Active galactic nuclei -------------------------

    def get_AGNs(self, param, image_caract, sig_psf, supersamp=5, trunc=1.68):
        """Adds AGN images in the image plane"""

        # Parameters
        m_AGN, x0_AGN, y0_AGN = param[-3:]
        A_AGN = magn2amp(m_AGN)
        xim, yim, magn, fermat_pot = image_caract

        # AGN kernel
        npix_kernel = int(round(trunc / self.dimpix))
        if npix_kernel % 2 == 0:
            npix_kernel += 1

        # AGN supersampled kernel
        npix_supersamp = int(npix_kernel * supersamp)
        dimpix_supersamp = self.dimpix / supersamp
        xkern = (np.arange(npix_supersamp) - (npix_supersamp - 1) / 2) * dimpix_supersamp
        ykern = (np.arange(npix_supersamp) - (npix_supersamp - 1) / 2) * dimpix_supersamp
        kernel = np.asarray(np.meshgrid(xkern, ykern))
        inddown = int(np.floor(npix_supersamp / 2))
        indup = int(np.ceil(npix_supersamp / 2))

        # AGN normalized image
        AGN_kernel = gauss(kernel[0], kernel[1], 1., sig_psf, 0., 0.)
        AGN_kernel /= np.sum(AGN_kernel)

        # Image plane supersampled grid
        npix_grid = int(self.npix * supersamp)
        AGN_grid = np.zeros((npix_grid, npix_grid))

        for i in range(len(xim)):
            # AGN image
            AGN_im = AGN_kernel * A_AGN * abs(magn[i])

            # AGN pixel position in the image plane
            indx = xim[i] / self.dimpix + self.npix / 2
            indy = yim[i] / self.dimpix + self.npix / 2
            # AGn pixel position in the supersampled grid
            indx_supersamp = indx * supersamp + int(supersamp / 2)
            indy_supersamp = indy * supersamp + int(supersamp / 2)

            # Correction of the shift between the discrete and the exact positions
            indx_super_int = int(round(indx_supersamp))
            indy_super_int = int(round(indy_supersamp))
            xshift = indx_super_int - indx_supersamp
            yshift = indy_super_int - indy_supersamp
            AGN_shift = interpolation.shift(AGN_im, [-yshift, -xshift], order=1)

            # Selection of the supersampled grid section
            indx_min_g = np.maximum(0, indx_super_int - inddown)
            indx_max_g = np.minimum(npix_grid, indx_super_int + indup)
            indy_min_g = np.maximum(0, indy_super_int - inddown)
            indy_max_g = np.minimum(npix_grid, indy_super_int + indup)

            # Selection of the supersampled kernel section
            indx_min_k = np.maximum(0, -indx_super_int + inddown)
            indx_max_k = np.minimum(npix_supersamp, -indx_super_int + inddown + npix_grid)
            indy_min_k = np.maximum(0, -indy_super_int + inddown)
            indy_max_k = np.minimum(npix_supersamp, -indy_super_int + inddown + npix_grid)

            # Adding AGN to the supersampled grid
            AGN_grid[indy_min_g:indy_max_g, indx_min_g:indx_max_g] += AGN_shift[indy_min_k:indy_max_k,
                                                                      indx_min_k:indx_max_k]

        # Rebinning the supersampled grid
        AGN_bin = np.mean(AGN_grid.reshape(self.npix, supersamp, self.npix, supersamp), axis=(1, 3))

        return AGN_bin * supersamp ** 2

    def fig_AGN(self, im, param):

        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]
        m_AGN, x0_AGN, y0_AGN = param[-3:]

        plt.figure()
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")
        plt.imshow(im, extent=[-self.dim / 2, self.dim / 2, -self.dim / 2, self.dim / 2],
                   origin='lower', cmap='hot')  # , norm=LogNorm()
        plt.title(r"Lens : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_lens, y0_lens) +
                  r"f={:.1f}, $\theta$$_E$={:.1f}, ".format(ellip, theta_E) +
                  r"$\phi$={:.2f}$\pi$".format(phi / np.pi) + "\n" +
                  r"AGN : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_AGN, y0_AGN) +
                  r"m={:.1f}".format(m_AGN))
        cbar = plt.colorbar()
        cbar.set_label(r'CPS', rotation=270, fontsize=10,
                       labelpad=15)
        plt.clim(np.min(im), np.max(im))

    def fig_total(self, im, param):

        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]
        m_host, sig_host, x0_host, y0_host, m_AGN, x0_AGN, y0_AGN = param[7:]

        plt.figure()
        plt.xlabel("x (arcsec)")
        plt.ylabel("y (arcsec)")
        plt.imshow(im, extent=[-self.dim / 2, self.dim / 2, -self.dim / 2, self.dim / 2],
                   origin='lower', cmap='hot')  # , norm=LogNorm()
        plt.title(r"Lens : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_lens, y0_lens) +
                  r"f={:.1f}, $\theta$$_E$={:.1f}, ".format(ellip, theta_E) +
                  r"$\phi$={:.2f}$\pi$".format(phi / np.pi) + "\n" +
                  r"Host : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_host, y0_host) +
                  r"m={:.1f}, $\sigma$={:.2f}".format(m_host, sig_host) + "\n" +
                  r"AGN : x$_0$={:.2f}, y$_0$={:.2f}, ".format(x0_AGN, y0_AGN) +
                  r"m={:.1f}".format(m_AGN))
        cbar = plt.colorbar()
        cbar.set_label(r'CPS', rotation=270, fontsize=10,
                       labelpad=15)
        plt.clim(np.min(im), np.max(im))

    # ------------------------------- Time delays ----------------------------------

    def get_time_delays(self, cosmo, image_caract):
        """Time delays between AGN images"""
        # Parameters
        zs, zd, Ds, Dd, Dds, vdisp, H0 = cosmo
        xim, yim, magn, fermat_pot = image_caract

        # Time delays
        time_delays = (1 + zd) * Dd * Ds / Dds / c.value * (fermat_pot - np.min(fermat_pot))
        time_delays *= (2 * np.pi / 360 / 3600) ** 2 # Conversion to days

        if len(fermat_pot) == 2:
            pad = -np.ones((2))
            time_delays = np.concatenate((time_delays, pad), axis=None)

        return time_delays

    # ----------------------------------- Tests ------------------------------------

    def cuts_and_caustics(self, param):
        """Returns distance lens-host and distance lens-cuts at the same angle"""

        # Parameters
        x0_lens, y0_lens, ellip, phi, theta_E, gamma_ext, phi_ext = param[:7]
        gamma1, gamma2 = pol2cart(gamma_ext, phi_ext)
        x0_AGN, y0_AGN = param[-2:]
        f = ellip
        f_prime = np.sqrt(1 - f ** 2)
        rad = np.linspace(-np.pi, np.pi - (2 * np.pi / 1000), 1000)

        # AGN position relative to the lens
        r_AGN, phi_AGN = cart2pol(x0_AGN - x0_lens, y0_AGN - y0_lens)

        # Cuts in lens coordinate system
        cutx = -theta_E * np.sqrt(f) / f_prime * np.arcsin(f_prime * np.sin(rad))
        cuty = -theta_E * np.sqrt(f) / f_prime * np.arcsinh(f_prime / f * np.cos(rad))
        cut_vec = np.array([cutx, cuty])
        # Cuts in fiel of view coordinates
        cut_rot = np.tensordot(mtx_rot(phi), cut_vec, axes=1)
        r_cut, phi_cut = cart2pol(cut_rot[0], cut_rot[1])
        # Distance cut-lens radially aligned with the source
        ind_cut = np.where(abs(phi_cut - phi_AGN) == np.min(abs(phi_cut - phi_AGN)))  # !!! Interpolation à la place ???
        dcut = r_cut[ind_cut]

        # Caustics  in lens coordinate system
        delta = norm(np.cos(rad), f * np.sin(rad))
        r = theta_E * np.sqrt(f) / delta
        caustx = r * np.sin(rad) + cutx - gamma1 * r * np.sin(rad) - gamma2 * r * np.cos(rad)
        causty = r * np.cos(rad) + cuty - gamma2 * r * np.sin(rad) + gamma1 * r * np.cos(rad)
        caust_vec = np.array([caustx, causty])
        # Caustics in fiel of view coordinates
        caust_rot = np.tensordot(mtx_rot(phi), caust_vec, axes=1)
        r_caust, phi_caust = cart2pol(caust_rot[0], caust_rot[1])
        # Distance caust-lens radially aligned with the source
        ind_caust = np.where(
            abs(phi_caust - phi_AGN) == np.min(abs(phi_caust - phi_AGN)))  # !!! Interpolation à la place ???
        dcaust = r_caust[ind_caust]

        # !!! Égalités?
        if r_AGN > dcut and r_AGN > dcaust:
            nbr_im = 1
        if r_AGN < dcut and r_AGN > dcaust:
            nbr_im = 2
        if r_AGN > dcut and r_AGN < dcaust:
            nbr_im = 3
        if r_AGN < dcut and r_AGN < dcaust:
            nbr_im = 4

        return nbr_im, cut_rot, caust_rot

    def plot_cuts_and_caustics(self, param):

        x0_lens, y0_lens, ellip, phi, theta_E = param[:5]
        x0_AGN, y0_AGN = param[-2:]
        nbr_im, cut_rot, caust_rot = self.cuts_and_caustics(param)
        plt.scatter(cut_rot[0] + x0_lens, cut_rot[1] + y0_lens, s=.1, c='k')
        plt.scatter(caust_rot[0] + x0_lens, caust_rot[1] + y0_lens, s=.1, c='k')
        plt.plot(x0_AGN, y0_AGN, 'og')
        plt.axis('equal')

    # ------------------------------ Training examples -----------------------------

    def generate(self, fig=np.zeros((8)), savefig=np.zeros((8)), save=True, test=False):
        """Generates training examples"""

        # Opening files
        if save:
            file = h5py.File(self.path + 'dataset.hdf5', 'a')
            set_im = file.create_dataset("images", (self.nsamp, 1, self.npix, self.npix), dtype='f')
            set_param = file.create_dataset("parameters", (self.nsamp, 14), dtype='f')
            set_shft = file.create_dataset("redshifts", (self.nsamp, 2), dtype='f')
            set_dt = file.create_dataset("time_delays", (self.nsamp, 4), dtype='f')
            set_pot = file.create_dataset("Fermat_potential", (self.nsamp, 4), dtype='f')
            set_pos = file.create_dataset("positions", (self.nsamp, 2, 4), dtype='f')
            set_H0 = file.create_dataset("Hubble_cst", (self.nsamp, 1), dtype='f')


        it = 0  # initialisation of iterations
        while it < self.nsamp:  # Loop on examples

            # Drawing parameters from prior distributions

            ### Cosmology
            # Hubble constant (km/s/Mpc)
            H0 = np.random.uniform(64., 76.)
            # Cosmological model
            cosmo_model = FlatLambdaCDM(H0=H0, Om0=.3)

            ### Einstein radius
            # Source redshift
            zs = 1.5  # np.random.uniform(1.,4.) # 1.,3.
            # Lens redshift
            zd = .5  # np.random.uniform(.04,1.) # .04,.5
            # Source-observer angular diameter (Mpc)
            Ds = cosmo_model.angular_diameter_distance(zs)
            # Lens-observer angular diameter (Mpc)
            Dd = cosmo_model.angular_diameter_distance(zd)
            # Source-lens angular diameter (Mpc)
            Dds = cosmo_model.angular_diameter_distance_z1z2(zd, zs)
            # Velocity dispersion (Mpc/day)
            vdisp = (np.random.uniform(225., 275.) * u.km / u.s).to('Mpc/d')  # 150,300 ou fixe 300
            # Einstein radius (arcsec)
            theta_E = 4 * np.pi * (vdisp / c) ** 2 * Dds / Ds * 180 * 3600 / np.pi
            if theta_E < .5 or theta_E > 2.:  # .4 or 2.5
                print('Einstein radius out of bounds : {}'.format(theta_E))
                continue

            ### Lens (SIE)
            # Lens x coordinate (arcsec)
            x0_lens = np.random.uniform(-.3,.3)  # -.05
            # Lens y coordinate (arcsec)
            y0_lens = np.random.uniform(-.3,.3)  # .02
            # Lens ellipticity
            ellip = np.random.uniform(.3, .99)  # .7
            # Lens inclination (rad)
            phi = np.random.uniform(-np.pi/2,np.pi/2)  # np.pi / 3

            ### External shear
            # External shear amplitude
            gamma_ext = np.random.uniform(0.,.05)  # .03
            # External shear angle (rad)
            phi_ext = np.random.uniform(-np.pi/2,np.pi/2)  # np.pi / 4

            ### Gaussian host galaxy
            # Host galaxy magnitude (AB system)
            # m_host=np.random.uniform(25,20)
            m_host = 22.5
            # Host galaxy extent (arcsec)
            # sig_host=np.random.uniform(.15,.35)
            sig_host = .15  # .25
            # Host x coordinate (arcsec)
            # x0_host=np.random.uniform(-.7,.7)
            x0_host = -.03125  # -.35
            # Host y coordinate (arcsec)
            # y0_host=np.random.uniform(-.7,.7)
            y0_host = .03125  # .35

            ### Active Galactic Nuclei (point source)
            # AGN magnitude (AB system)
            # m_AGN=np.random.uniform(22.5,20)
            m_AGN = 21.25
            # AGN x coordinate (arcsec)
            # x0_AGN = x0_host + np.random.uniform(-.25*sig_host,.25*sig_host)
            x0_AGN = 0  # -.31875
            # AGN y coordinate (arcsec)
            # y0_AGN = y0_host + np.random.uniform(-.25*sig_host,.25*sig_host)
            y0_AGN = 0  # .38125

            param = np.array([x0_lens, y0_lens, ellip, phi, theta_E,
                              gamma_ext, phi_ext,
                              m_host, sig_host, x0_host, y0_host,
                              m_AGN, x0_AGN, y0_AGN])

            cosmo = np.array([zs, zd, Ds.value, Dd.value,
                              Dds.value, vdisp.value, H0])

            # There must be two or four AGN images
            xim, yim, magn, fermat_pot = self.AGN_images(param)
            image_caract = [xim, yim, magn, fermat_pot]
            if (len(xim) != 2) and (len(xim) != 4):
                print("Wrong number of images : {}".format(len(xim)))
                continue

            # Compute kappa_map, deflection angles, host galaxy light profil,
            # host galaxy lensed image, convolve PSF, add AGN images and compute
            # time delays
            kappa_map = None
            host = self.gaussian_host(param)
            if self.analytique:
                alpha = self.SIE_alpha(param)
            else:
                alpha = self.alpha_convolution(kappa_map)
            alpha += self.external_shear(param)
            beta, im_host = self.host_image(alpha, host)
            im_psf = self.gaussian_PSF(im_host, self.sig_psf)
            im_AGN = self.get_AGNs(param, image_caract, self.sig_psf)
            im = im_psf + im_AGN
            time_delays = self.get_time_delays(cosmo, image_caract)

            # Tests
            # Lensed image must be inside the field of view
            if np.max(abs(beta)) >= self.dim / 2:
                print("Beta out of bound : {}".format(np.max(abs(beta))))
                continue
            # There shouldn't be any NaN values
            if np.sum(np.isnan(im)) != 0:
                print('NaN values : {}'.format(np.sum(np.isnan(im))))
                continue

            # Figures
            if fig[0]:
                self.fig_kappa(kappa_map, param, test, )
                if savefig[0]:
                    plt.savefig(self.path + "kappa{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[1]:
                self.fig_alpha(alpha, param)
                if savefig[1]:
                    plt.savefig(self.path + "alpha_ana{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[2]:
                self.fig_alpha(alpha, param)
                if savefig[2]:
                    plt.savefig(self.path + "alpha_conv{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[3]:
                self.fig_host(host, param)
                if savefig[3]:
                    plt.savefig(self.path + "host{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[4]:
                self.fig_image(im_host, param)
                if savefig[4]:
                    plt.savefig(self.path + "im_host{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[5]:
                self.fig_image(im_psf, param)
                if savefig[5]:
                    plt.savefig(self.path + "host_psf{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[6]:
                self.fig_AGN(im_AGN, param)
                if savefig[6]:
                    plt.savefig(self.path + "im_AGN{}.png".format(it), dpi=600, bbox_inches='tight')
            if fig[7]:
                self.fig_total(im, param)
                if savefig[7]:
                    plt.savefig(self.path + "im{}.png".format(it), dpi=600, bbox_inches='tight')

            # Saving
            if save:
                set_im[it, :] = im.reshape(1, self.npix, self.npix)
                set_param[it, :] = param
                set_shft[it, :] = np.array([zd, zs])
                set_dt[it, :] = time_delays
                if len(fermat_pot) == 2:
                    pad = -np.ones((2))
                    fermat_pot = np.concatenate((fermat_pot - np.min(fermat_pot), pad), axis=None)
                    xim = np.concatenate((xim, pad), axis=None)
                    yim = np.concatenate((yim, pad), axis=None)
                    set_pot[it, :] = fermat_pot
                else:
                    set_pot[it, :] = fermat_pot - np.min(fermat_pot)
                set_pos[it, :] = np.array([xim, yim])
                set_H0[it, :] = np.array([H0])

            it += 1  # update iteration

        # Fermeture des fichiers
        if save:
            file.close()
