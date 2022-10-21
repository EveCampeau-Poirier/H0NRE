# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import median_absolute_error as med_abs_err

# PyTorch libraries
import torch
from torch.nn.utils import clip_grad_norm_
from torch import autograd

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# --- Functions for training -----------------------------------------------

def ellip_coordinates(f, phi):
    e1 = (1 - f) / (1 + f) * np.cos(2 * phi)
    e2 = (1 - f) / (1 + f) * np.sin(2 * phi)
    return e1, e2


def shear_coordinates(gamma, phi):
    gamma1 = gamma * np.cos(2 * phi)
    gamma2 = gamma * np.sin(2 * phi)
    return gamma1, gamma2


def ellip_polar(e1, e2):
    x = np.sqrt(e1 ** 2 + e2 ** 2)
    f = (1 - x) / (1 + x)
    phi = np.arctan2(e2, e1) / 2
    return f, phi


def shear_polar(gamma1, gamma2):
    gamma = np.sqrt(gamma1 ** 2 + gamma2 ** 2)
    phi = np.arctan2(gamma2, gamma1) / 2
    return gamma, phi


def linear(x, a, b):
    return a * x + b


# NORM
def norm(x, y):
    return torch.sqrt(x ** 2 + y ** 2)


# ROTATION MATRIX
def mtx_rot(phi):
    mtx = torch.zeros(phi.shape[0], 1, 2, 2)
    mtx[:, 0, 0, 0] = torch.cos(phi)
    mtx[:, 0, 0, 1] = -torch.sin(phi)
    mtx[:, 0, 1, 0] = torch.sin(phi)
    mtx[:, 0, 1, 1] = torch.cos(phi)
    return mtx


# CARTESIAN COORDINATES TO SPHERICAL
def cart2pol(x, y):
    return norm(x, y), torch.arctan2(y, x)


# SPHERICAL COORDINATES TO CARTESIAN
def pol2cart(r, theta):
    return r * torch.cos(theta), r * torch.sin(theta)


def normalization(x, y):
    """
        Normalizes distributions
        Inputs
            x : (array) H0 values
            y : (array) posterior probabilities
        Outputs
            y : (array) Norm
        """
    norm = np.trapz(y, x, axis=1)
    y /= norm[:, None]

    return y


def gaussian_noise(x, sig_dt=.3):
    """
    Adds noise to time delays
    Inputs
        x : (tensor)[batch_size x 4 x 2] Time delays and Fermat potentials
        sig_dt : (float) noise standard deviation on time delays
        sig_pot : (float) noise standard deviation on potentials
    Outputs
        noisy_data : (tensor)[batch_size x 4 x 2] noisy time delays + true Fermat potential
    """
    mask_zero = torch.where(x == 0, 0, 1)
    mask_pad = torch.where(x == -1, 0, 1)
    noise_dt = sig_dt * torch.randn((x.size(0), x.size(1)))
    noisy_data = x.clone()
    noisy_data += noise_dt * mask_zero * mask_pad

    return noisy_data


def r_estimator(model, x1, x2):
    """
    Likelihood ratio estimator
    Inputs
        x1 : (tensor)[nexamp x 1 x npix x npix] time delays mask
        x2 : (tensor)[nexamp x 1] Hubble constant
    Outputs
        lr : (array)[nexamp] likelihood ratio
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x1 = x1.to(device, non_blocking=True).float()
    x2 = x2.to(device, non_blocking=True).float()

    model = model.to(device, non_blocking=True)
    model.eval()

    with torch.no_grad():
        output = model(x1, x2)

    sm = torch.nn.Softmax(dim=1)
    prob = sm(output)
    s = prob[:, 1].detach().cpu().numpy()
    lr = s / (1 - s)

    return lr


def get_Fermat_potentials(x0_lens, y0_lens, theta_E, ellip, phi, gamma_ext, phi_ext, xim_fov, yim_fov, x0_AGN=0.,
                          y0_AGN=0.):
    """Computes the position and the magnification of AGN images
    Also outputs the Fermat potential at these positions"""

    # Parameters
    theta_E = theta_E.unsqueeze(1)
    f = ellip.unsqueeze(1)
    f_prime = np.sqrt(1 - f ** 2)

    # Image positions in the lens coordinate system
    im_pos_trans = torch.zeros(xim_fov.shape[0], xim_fov.shape[1], 2, 1)
    im_pos_trans[:, :, 0, 0] = xim_fov - x0_lens.unsqueeze(1)
    im_pos_trans[:, :, 1, 0] = yim_fov - y0_lens.unsqueeze(1)
    im_pos_lens = torch.matmul(mtx_rot(-phi), im_pos_trans).squeeze(3)
    xim, yim = im_pos_lens[:, :, 0], im_pos_lens[:, :, 1]
    r_im, phi_im = cart2pol(yim, xim)

    # AGN positions in the lens coordinate system
    src_pos_trans = torch.zeros(x0_lens.shape[0], 1, 2, 1)
    src_pos_trans[:, 0, 0, 0] = x0_AGN - x0_lens
    src_pos_trans[:, 0, 1, 0] = y0_AGN - y0_lens
    src_pos_lens = torch.matmul(mtx_rot(-phi), src_pos_trans).squeeze(3)
    xsrc, ysrc = src_pos_lens[:, :, 0], src_pos_lens[:, :, 1]

    # Coordinate translation for the external shear
    lens_pos = torch.cat((x0_lens.unsqueeze(1), y0_lens.unsqueeze(1)), dim=1).unsqueeze(1).unsqueeze(3)
    lens_pos_rot = torch.matmul(mtx_rot(-phi), lens_pos).squeeze(3)
    xtrans, ytrans = lens_pos_rot[:, :, 0], lens_pos_rot[:, :, 1]

    # External shear in the lens coordinate system
    gamma1_fov, gamma2_fov = pol2cart(gamma_ext, phi_ext)
    gamma_vec = torch.cat((gamma1_fov.unsqueeze(1), gamma2_fov.unsqueeze(1)), dim=1).unsqueeze(1).unsqueeze(3)
    gamma_vec_rot = torch.matmul(mtx_rot(-2 * phi), gamma_vec).squeeze(3)
    gamma1, gamma2 = gamma_vec_rot[:, :, 0], gamma_vec_rot[:, :, 1]

    # X deflection angle (eq. 27a Kormann et al. 1994)
    def alphax(varphi):
        return theta_E * torch.sqrt(f) / f_prime * torch.arcsin(f_prime * torch.sin(varphi))

    # Y deflection angle (eq. 27a Kormann et al. 1994)
    def alphay(varphi):
        return theta_E * torch.sqrt(f) / f_prime * torch.arcsinh(f_prime / f * torch.cos(varphi))

    # Lens potential deviation from a SIS (Meneghetti, 19_2018, psi_tilde)
    def psi_tilde(varphi):
        return torch.sin(varphi) * alphax(varphi) + torch.cos(varphi) * alphay(varphi)

    # Lens potential (eq. 26a Kormann et al. 1994)
    def lens_pot(varphi):
        return psi_tilde(varphi) * radial(varphi)

    # Shear potential (eq. 3.80 Meneghetti)
    def shear_pot(x, y):
        return gamma1 / 2 * (x ** 2 - y ** 2) + gamma2 * x * y

    # Radial componant (eq. 34 Kormann et al. 1994)
    def radial(varphi):
        usual_rad = ysrc * torch.cos(varphi) + xsrc * torch.sin(varphi) + psi_tilde(varphi)
        translation = gamma1 * (xtrans * torch.sin(varphi) - ytrans * torch.cos(varphi)) + gamma2 * (
                xtrans * torch.cos(varphi) + ytrans * torch.sin(varphi))
        shear_term = 1 + gamma1 * (torch.cos(varphi) ** 2 - torch.sin(varphi) ** 2) - 2 * gamma2 * torch.cos(
            varphi) * torch.sin(varphi)
        return (usual_rad + translation) / shear_term

    # Fermat potential
    geo_term = norm(xim - xsrc, yim - ysrc) ** 2 / 2
    lens_term = lens_pot(phi_im)
    shear_term = shear_pot(xim + xtrans, yim + ytrans)
    fermat_pot = geo_term - lens_term - shear_term
    fermat_pot = fermat_pot - torch.amin(fermat_pot, dim=1, keepdim=True)

    return fermat_pot


def acc_fct(y_hat, y):
    """
    Computes the model's predictions accuracy
    Inputs
        y_hat : (tensor) [batch_size x 2] Model's prediction
        y : (tensor) [batch_size] Labels
    Outputs
        acc : (float) Accuracy
    """
    diff = torch.abs(y_hat - y)
    threshold = .05
    acc = torch.mean((diff <= threshold * torch.abs(y)).float())
    return acc


def split_data(path_in):
    """
    Reads data, generates classes and splits data into training, validation and test sets
    Inputs
        file : (str) name of the file containing data
        path_in : (str) file's directory
    Outputs
        All outputs are a list of the time delays, the Fermat potential, the H0 and the labels tensors
        train_set : (list) training set. Corresponds to 80% of all data.
        valid_set : (list) validation set. Corresponds to 10% of all data.
    """
    # Reading data
    dataset = h5py.File(os.path.join(path_in, "dataset.hdf5"), 'r')
    images = dataset["images"][:]
    param = dataset["parameters"][:, :7]
    dataset.close()
    
    images = images / np.amax(images, axis=(1,2,3), keepdims=True)

    param[:, 2], param[:, 3] = ellip_coordinates(param[:, 2], param[:, 3])
    param[:, 5], param[:, 6] = shear_coordinates(param[:, 5], param[:, 6])

    mean = np.mean(param, axis=0)
    std = np.std(param, axis=0)
    param = (param - mean) / std

    # import keys
    keys = h5py.File(os.path.join(path_in, "keys.hdf5"), 'r')
    train_keys = keys["train"][:]
    valid_keys = keys["valid"][:]
    keys.close()

    # Splitting sets
    x_train, y_train = images[train_keys], param[train_keys]
    x_valid, y_valid = images[valid_keys], param[valid_keys]

    # Outputs
    train_set = [torch.from_numpy(x_train), torch.from_numpy(y_train)]
    valid_set = [torch.from_numpy(x_valid), torch.from_numpy(y_valid)]

    return train_set, valid_set


def train_fn(model, path_in, path_out, optimizer, loss_fn, acc_fn, threshold, sched=None,
             grad_clip=None, anomaly_detection=False, batch_size=128, epochs=100, std_noise=.04):
    """
    Manages the training
    Inputs
        model : (module object) neural network
        file : (str) name of the file containing data
        path_in : (str) path to data
        path_out : (str) path where the trained model and the results will be saved
        optimizer : (torch optimizer) optimizer
        loss_fn : (object) training objective
        acc_fn : (function) metric to evaluate accuracy
        sched : (function?) If not None, learning rate scheduler to update
        threshold : (int) Epoch at which the scheduler stops updating the learning rate
        grad_clip : (float) If not None, max norm for gradient clipping
        anomaly_detection : (bool) If True, the forward pass is done with autograd.detect_anomaly()
        batch_size : (int) batch size
        epochs : (int) number of epochs
    Outputs
        None
    """

    # model and loss function on GPU
    model = model.to(device, non_blocking=True)
    loss_fn = loss_fn.to(device, non_blocking=True)

    # Datasets
    train_set, valid_set = split_data(path_in)
    x_train, y_train = train_set
    x_val, y_val = valid_set

    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataset_valid = torch.utils.data.TensorDataset(x_val, y_val)

    # File to save logs
    if os.path.isfile(path_out + '/logs.hdf5'):
        os.remove(path_out + '/logs.hdf5')
    save_file = h5py.File(path_out + '/logs.hdf5', 'a')

    trng = save_file.create_group("training")
    train_loss = trng.create_dataset("loss", (epochs, 1), dtype='f')
    train_acc = trng.create_dataset("accuracy", (epochs, 1), dtype='f')

    vldt = save_file.create_group("validation")
    valid_loss = vldt.create_dataset("loss", (epochs, 1), dtype='f')
    valid_acc = vldt.create_dataset("accuracy", (epochs, 1), dtype='f')

    # starting timer
    start = time.time()

    # loop on epochs
    for epoch in range(epochs):

        # Printing current epoch
        print('\n')
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # loop on validation and training phases
        for phase in ['valid', 'train']:
            if phase == 'train':
                model.train(True)  # training
                dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
            else:
                model.train(False)  # evaluation
                dataloader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size)

            # Initialization of cumulative loss on batches
            running_loss = 0.0
            # Initialization of cumulative accuracy on batches
            running_acc = 0.0

            step = 0  # step initialization (batch number)
            # loop on batches
            for x, y in dataloader:
                x += std_noise * torch.randn(x.shape)
                x = x.float().cuda()
                y = y.float().cuda()

                # training phase
                if phase == 'train':
                    # Forward pass
                    for param in model.parameters():
                        param.grad = None

                    if anomaly_detection:
                        with autograd.detect_anomaly():
                            y_hat = model(x)
                            loss = loss_fn(y_hat, y)
                    else:
                        y_hat = model(x)
                        loss = loss_fn(y_hat, y)

                    # Backward Pass
                    loss.backward()

                    if grad_clip is not None:
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    optimizer.step()

                # validation phase
                else:
                    with torch.no_grad():
                        y_hat = model(x)
                        loss = loss_fn(y_hat, y)

                # accuracy evaluation
                acc = acc_fn(y_hat, y)

                # update cumulative values
                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                # print current information
                if step % int(len(dataloader.dataset) / batch_size / 20) == 0:  #
                    print(
                        f'Current {phase} step {step} ==>  Loss: {float(loss):.4e} // Acc: {float(acc):.4e} // AllocMem (Gb): {torch.cuda.memory_reserved(0) * 1e-9}')

                step += 1

            # mean of loss and accuracy on epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            # print means
            print(f'{phase} Loss: {float(epoch_loss):.4e} // Acc: {float(epoch_acc):.4e}')

            # append means to lists
            if phase == 'train':
                train_loss[epoch, :] = epoch_loss.detach().cpu().numpy()
                train_acc[epoch, :] = epoch_acc.detach().cpu().numpy()
                if sched is not None and epoch <= threshold:
                    sched.step()
            else:
                valid_loss[epoch, :] = epoch_loss.detach().cpu().numpy()
                valid_acc[epoch, :] = epoch_acc.detach().cpu().numpy()

        # Keeping track of the model
        if epoch % int(epochs / 10) == 0:
            torch.save(model, path_out + f'/models/model{epoch:02d}.pt')

    # Closing file
    save_file.close()

    # print training time
    time_elapsed = time.time() - start
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')


# --- plots ----------------------------------------------
def learning_curves(file, path_out):
    """
    Plots learning curves
    Inputs
        file : (str) name of the logs file
        path_out : (str) directory to save plots
    Outputs
        None
    """

    # data extraction
    data = h5py.File(file, 'r')
    train_loss = data["training/loss"][:]
    valid_loss = data["validation/loss"][:]
    train_acc = data["training/accuracy"][:]
    valid_acc = data["validation/accuracy"][:]
    data.close()

    # loss
    plt.figure()
    plt.plot(train_loss, 'b', label='Training')
    plt.plot(valid_loss, 'r', label='Validation')
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.legend()
    plt.savefig(path_out + '/loss.png', bbox_inches='tight')

    # accuracy
    plt.figure()
    plt.plot(100 * train_acc, 'b', label='Training')
    plt.plot(100 * valid_acc, 'r', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.legend()
    plt.savefig(path_out + '/acc.png', bbox_inches='tight')


def modeling(file_keys, file_data, file_model, path_out, nrow=2, ncol=4, std_noise=.04):
    """
    Inputs
        file_keys : (str) name of the file containing the keys of the test set
        file_data : (str) name of the file containing data
        file_model : (str) name of the file containing the model
        path_out : (str) directory where to save the output
    Outputs
        None
    """
    if torch.cuda.is_available():
        model = torch.load(file_model)
    else:
        model = torch.load(file_model, map_location="cpu")

    # import keys
    keys = h5py.File(file_keys, 'r')
    test_keys = keys["test"][:]
    keys.close()

    # import data
    dataset = h5py.File(file_data, 'r')
    images = dataset["images"][test_keys]
    param = dataset["parameters"][:, :7]
    dataset.close()
    
    images = images / np.amax(images, axis=(1,2,3), keepdims=True)
    param[:, 2], param[:, 3] = ellip_coordinates(param[:, 2], param[:, 3])
    param[:, 5], param[:, 6] = shear_coordinates(param[:, 5], param[:, 6])
    mean = np.mean(param, axis=0)
    std = np.std(param, axis=0)
    truths = param[test_keys]

    x = torch.from_numpy(images) + std_noise * torch.randn(images.shape)
    x = x.to(device, non_blocking=True).float()
    y = torch.from_numpy(truths).to(device, non_blocking=True).float()
    model = model.to(device, non_blocking=True)
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    acc = acc_fct(y_hat, y)
    print('\n Accuracy on test set : ', acc * 100)

    # Conversion en format numpy et redimensionner
    predic = y_hat.detach().cpu().numpy() * std + mean

    names = [r"$x_d$", r"$y_d$", r"$e_x$", r"$e_y$",
             r"$\theta_E$", r"$\gamma_x$", r"$\gamma_y$"]

    mae = med_abs_err(truths, predic, multioutput='raw_values')
    print("\n Median absolute error")
    for i in range(len(names)):
        print(names[i], ' : ', mae[i])

    ticks = np.array([[-.4, .6, .2], [-.4, .6, .2], [-1., 1., .5], [-1., 1., .5],
                      [.5, 2.5, .5], [-.07, .12, .05], [-.07, .12, .05]])

    it = 0
    fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    for i in range(nrow):
        for j in range(ncol):
            if it == 7:
                axs[i, j].set_visible(False)
            else:
                upper = np.maximum(np.max(truths[:, it]), np.max(predic[:, it]))
                lower = np.minimum(np.min(truths[:, it]), np.min(predic[:, it]))
                x = np.linspace(lower, upper, 1000)
                axs[i, j].scatter(truths[:, it], predic[:, it], s=.1, c='k', alpha=.4)
                axs[i, j].plot(x, x, '-g')
                axs[i, j].set_title(names[it])
                axs[i, j].axis('equal')
                axs[i, j].xaxis.set_ticks(np.arange(ticks[it, 0], ticks[it, 1], ticks[it, 2]))
                axs[i, j].yaxis.set_ticks(np.arange(ticks[it, 0], ticks[it, 1], ticks[it, 2]))

                opt, cov = curve_fit(linear, truths[:, it], predic[:, it])
                reg = linear(x, opt[0], opt[1])
                if opt[1] < 0:
                    axs[i, j].plot(x, reg, '-r', label=r"{:.2f}$x-${:.2f}".format(opt[0], abs(opt[1])))
                else:
                    axs[i, j].plot(x, reg, '-r', label=r"{:.2f}$x+${:.2f}".format(opt[0], opt[1]))
                diff = linear(truths[:, it], opt[0], opt[1]) - predic[:, it]
                # 1,96 sigma = 97,5e centile de la distribution normale standard = intervalle de confiance de 95%
                err = 1.96 * diff.std()
                axs[i, j].fill_between(x, reg - err, reg + err, alpha=0.2, facecolor='r',
                                       label="± {:.2f}".format(err))
                axs[i, j].legend(loc='upper left', frameon=False)
                it += 1

    plt.subplots_adjust(left=.125, bottom=.1, right=.9,
                        top=.9, wspace=.25, hspace=.3)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("truth")
    plt.ylabel("prediction")
    plt.savefig(path_out + '/performance.png', bbox_inches='tight')

    return predic


def inference(param_pred, path_data, file_model, path_out, nrow=5, ncol=4, npts=1000, batch_size=100):
    """
    Computes the NRE posterior, the analytical posterior and performs the coverage diagnostic
    Inputs
        file_keys : (str) name of the file containing the keys of the test set
        file_data : (str) name of the file containing data
        file_model : (str) name of the file containing the model
        path_out : (str) directory where to save the output
    Outputs
        None
    """
    if torch.cuda.is_available():
        model = torch.load(file_model)
    else:
        model = torch.load(file_model, map_location="cpu")

    # import keys
    keys = h5py.File(os.path.join(path_data, "keys.hdf5"), 'r')
    test_keys = keys["test"][:]
    keys.close()

    # import data
    dataset = h5py.File(os.path.join(path_data, "dataset.hdf5"), 'r')
    H0_true = dataset["Hubble_cst"][test_keys]
    lower_bound = np.floor(np.min(H0_true))
    higher_bound = np.ceil(np.max(H0_true))

    # remove out of bounds data
    idx_up = np.where(H0_true > 75.)[0]
    idx_down = np.where(H0_true < 65.)[0]
    idx_out = np.concatenate((idx_up, idx_down))
    H0_true = np.delete(H0_true, idx_out, axis=0)
    H0_true = H0_true.flatten()
    test_keys = np.delete(test_keys, idx_out, axis=0)
    time_delays = dataset["time_delays"][test_keys]
    im_pos = torch.from_numpy(dataset["positions"][test_keys])
    dataset.close()

    param_pred[:, 2], param_pred[:, 3] = ellip_polar(param_pred[:, 2], param_pred[:, 3])
    param_pred[:, 5], param_pred[:, 6] = shear_polar(param_pred[:, 5], param_pred[:, 6])
    param_pred = torch.from_numpy(np.delete(param_pred, idx_out, axis=0))
    nsamp = time_delays.shape[0]
    dt = gaussian_noise(torch.from_numpy(time_delays))
    nim = torch.count_nonzero(dt + 1, dim=1)

    ind2 = torch.where(nim == 2)
    pos_doub = im_pos[ind2][:, :, :-2]
    param_doub = param_pred[ind2]
    pot_doub = get_Fermat_potentials(param_doub[:, 0], param_doub[:, 1], param_doub[:, 4],
                                     param_doub[:, 2], param_doub[:, 3], param_doub[:, 5],
                                     param_doub[:, 6], pos_doub[:, 0], pos_doub[:, 1])
    pot_doub = torch.cat((pot_doub, -torch.ones(pot_doub.shape[0], 2)), dim=1)
    
    ind4 = torch.where(nim == 4)
    pos_quad = im_pos[ind4]
    param_quad = param_pred[ind4]
    pot_quad = get_Fermat_potentials(param_quad[:, 0], param_quad[:, 1], param_quad[:, 4],
                                     param_quad[:, 2], param_quad[:, 3], param_quad[:, 5],
                                     param_quad[:, 6], pos_quad[:, 0], pos_quad[:, 1])
    
    pot = torch.ones(nsamp, 4)
    pot[ind2] = pot_doub
    pot[ind4] = pot_quad

    data = torch.cat((dt[:, :, None], pot[:, :, None]), dim=2)
    data = data[data != 0].reshape(nsamp, 3, 2)
    data_repeated = torch.repeat_interleave(data, npts, dim=0)

    # Global NRE posterior
    support = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    support_tile = np.tile(support, (nsamp, 1))
    support = support.flatten()

    dataset_test = torch.utils.data.TensorDataset(data_repeated, torch.from_numpy(support_tile))
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    ratios = []
    for x1, x2 in dataloader:
        preds = r_estimator(model, x1, x2)
        ratios.extend(preds)

    ratios = np.asarray(ratios).reshape(nsamp, npts)
    support_tile = support_tile.reshape(nsamp, npts)
    ratios = normalization(support_tile, ratios)

    # predictions
    arg_pred = np.argmax(ratios, axis=1)
    pred = support_tile[np.arange(nsamp), arg_pred]

    it = 0
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=False, figsize=(3 * ncol, 3 * nrow))
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].plot(support, ratios[it], '-b', label='{:.2f}'.format(pred[it]))
            axes[i, j].vlines(H0_true[it], np.min(ratios[it]), np.max(ratios[it]), colors='r',
                              linestyles='dotted', label='{:.2f}'.format(H0_true[it]))
            axes[i, j].legend(frameon=False, borderpad=.2, handlelength=.6, fontsize=9, handletextpad=.4)
            if float(nim[it]) == 4:
                axes[i, j].set_title("Quad")
            if float(nim[it]) == 2:
                axes[i, j].set_title("Double")
            # axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
            if i == int(nrow - 1):
                axes[i, j].set_xlabel(r"H$_0$ (km Mpc$^{-1}$ s$^{-1}$)")
            if j == 0:
                axes[i, j].set_ylabel(r"$p(H_0$ | $\Delta_t, \Delta_\phi)$")
            it += 1

    # saving
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.savefig(path_out + '/posteriors.png', bbox_inches='tight')

    # file to save
    if os.path.isfile(path_out + "/posteriors.hdf5"):
        os.remove(path_out + "/posteriors.hdf5")
    post_file = h5py.File(path_out + "/posteriors.hdf5", 'a')

    NRE = post_file.create_group("NRE_global")
    H0 = NRE.create_dataset("Hubble_cst", (npts,), dtype='f')
    post = NRE.create_dataset("posterior", (nsamp, npts), dtype='f')

    truth_set = post_file.create_dataset("truth", (nsamp,), dtype='f')

    H0[:] = support
    post[:, :] = ratios
    truth_set[:] = H0_true

    post_file.close()

    # Highest probability density region
    credibility = np.zeros((nsamp,))
    for i in range(nsamp):

        idx_truth = np.where(abs(support - H0_true[i]) == np.min(abs(support - H0_true[i])))[0]
        probs = ratios[i]
        prob_truth = probs[idx_truth]

        prob_false = probs.copy()
        prob_false[idx_truth] = 0.
        idx_equal_prob = np.where(abs(prob_false - prob_truth) == np.min(abs(prob_false - prob_truth)))[0]

        if len(idx_equal_prob) != 1:
            if idx_equal_prob[0] == idx_truth - 1 and idx_equal_prob[1] == idx_truth + 1:
                credibility[i] = 0
            continue

        start = np.minimum(idx_truth, idx_equal_prob)
        end = np.maximum(idx_truth, idx_equal_prob)
        idx_HPD = np.arange(start, end)

        credibility[i] = np.trapz(probs[idx_HPD], support[idx_HPD])

    # Coverage diagnostic
    bins = np.linspace(0., 1., 100)
    emperical_coverage = np.zeros_like(bins)
    for i in range(len(bins)):
        emperical_coverage[i] = np.mean(np.where(credibility <= bins[i], 1, 0))

    plt.style.use(['dark_background'])
    plt.figure()
    plt.plot(bins, bins, '--', color='white')
    plt.plot(bins, emperical_coverage, '-', color='lime')
    plt.xlabel("Credibility")
    plt.ylabel("Emperical coverage")
    plt.text(0., .9, "Underconfident", fontsize='large')
    plt.text(.65, .05, "Overconfident", fontsize='large')
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['savefig.facecolor'] = 'white'
    plt.savefig(path_out + '/coverage.png', bbox_inches='tight')