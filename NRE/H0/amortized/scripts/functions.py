# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science', 'bright'])

from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c

c = c.to('Mpc/d')  # Speed of light

# PyTorch libraries
import torch
from torch.nn.utils import clip_grad_norm_
from torch import autograd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)


# --- Functions for training -----------------------------------------------

# NORM
def norm(x, y):
    return torch.sqrt(x ** 2 + y ** 2)


# ROTATION MATRIX
def mtx_rot(phi):
    mtx = torch.zeros((phi.shape[0], 1, 2, 2), device=phi.device)
    mtx[:, 0, 0, 0] = torch.cos(phi)
    mtx[:, 0, 0, 1] = -torch.sin(phi)
    mtx[:, 0, 1, 0] = torch.sin(phi)
    mtx[:, 0, 1, 1] = torch.cos(phi)
    return mtx


# CARTESIAN COORDINATES TO SPHERICAL
def cart2pol(x, y):
    return norm(x, y), torch.atan2(y, x)


# SPHERICAL COORDINATES TO CARTESIAN
def pol2cart(r, theta):
    return r * torch.cos(theta), r * torch.sin(theta)


def get_Fermat_potentials(x0_lens, y0_lens, ellip, phi, theta_E, gamma_ext, phi_ext, xim_fov, yim_fov):
    """Computes the position and the magnification of AGN images
    Also outputs the Fermat potential at these positions"""

    # Parameters
    x0_AGN = 3e-6 * torch.randn((x0_lens.size(0)), device=x0_lens.device)
    y0_AGN = 3e-6 * torch.randn((y0_lens.size(0)), device=y0_lens.device)
    theta_E = theta_E.unsqueeze(1)
    f = ellip.unsqueeze(1)
    f_prime = torch.sqrt(1 - f ** 2)

    # Image positions in the lens coordinate system
    im_pos_trans = torch.zeros((xim_fov.shape[0], xim_fov.shape[1], 2, 1), device=xim_fov.device)
    im_pos_trans[:, :, 0, 0] = xim_fov - x0_lens.unsqueeze(1)
    im_pos_trans[:, :, 1, 0] = yim_fov - y0_lens.unsqueeze(1)
    im_pos_lens = torch.matmul(mtx_rot(-phi), im_pos_trans).squeeze(3)
    xim, yim = im_pos_lens[:, :, 0], im_pos_lens[:, :, 1]
    r_im, phi_im = cart2pol(yim, xim)

    # AGN positions in the lens coordinate system
    src_pos_trans = torch.zeros((x0_lens.shape[0], 1, 2, 1), device=x0_lens.device)
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


def doub_and_quad_potentials(nim, pos, param):

    pot = torch.tensor([[0], [0], [1]], device=pos.device)
    while torch.count_nonzero(pot) != pot.shape[0] * (pot.shape[1] - 1):

        param = param_noise(param)
        pos = pos_noise(pos)

        if torch.any(nim == 1):
            idd = torch.where(nim == 1)
            psd = pos[idd][:, :, :-2]
            prd = param[idd]
            potd = get_Fermat_potentials(prd[:, 0], prd[:, 1], prd[:, 2],
                                         prd[:, 3], prd[:, 4], prd[:, 5],
                                         prd[:, 6], psd[:, 0], psd[:, 1])
            potd = torch.cat((potd, -torch.ones((potd.shape[0], 2), device=potd.device)), dim=1)
            pot = potd

        if torch.any(nim == 3):
            idq = torch.where(nim == 3)
            psq = pos[idq]
            prq = param[idq]
            potq = get_Fermat_potentials(prq[:, 0], prq[:, 1], prq[:, 2],
                                         prq[:, 3], prq[:, 4], prq[:, 5],
                                         prq[:, 6], psq[:, 0], psq[:, 1])
            pot = potq

        if torch.any(nim == 1) and torch.any(nim == 3):
            pot = torch.ones((pos.shape[0], 4), device=potd.device)
            pot[idd] = potd
            pot[idq] = potq

    pot = pot[pot != 0.].view(pot.shape[0], -1)

    return pot


def dt_noise(dt, sig=.35):
    """
    Adds noise to time delays
    Inputs
        sig_dt : (float) noise standard deviation on time delays
    Outputs
        noisy_data : (tensor)[batch_size x 4 x 2] noisy time delays + true Fermat potential
    """
    mask = torch.where(dt > 0, 1, 0)
    noisy_dt = dt + sig * torch.randn((dt.size(0), dt.size(1)), device=dt.device) * mask

    return noisy_dt


def param_noise(param, sig=[4e-6, 4e-6, 1e-7, 4e-6, 4e-6, 4e-6, 4e-6]):
    """
    Adds noise to the parameters
    Inputs
        sig : (float) noise standard deviation on param
    Outputs
        noisy_param : (tensor)[batch_size x 7] noisy param
    """
    sig = torch.tensor(sig, device=param.device)
    noisy_param = param + sig[None, :] * torch.randn((param.size(0), param.size(1)), device=param.device)
    while torch.any(noisy_param[:, 2] >= 1):
        idx = torch.where(noisy_param[:, 2] >= 1)
        for i in idx[0]:
            new_noise = sig[2] * torch.randn((1), device=param.device)
            noisy_param[i, 2] = param[i, 2] + new_noise

    return noisy_param


def pos_noise(pos, sig=4e-6):
    """
        Adds noise to the image positions
        Inputs
            sig : (float) noise standard deviation on positions
        Outputs
            noisy_pos : (tensor)[batch_size x 2 x 4] noisy param
        """
    noisy_pos = pos + sig * torch.randn((pos.size(0), pos.size(1), pos.size(2)), device=pos.device)

    return noisy_pos


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


def log_trick(logp):
    """
    Compute the log-trick for numerical stability
    Inputs :
      log_p : log of a quantity to sum in a log
    Outputs :
      trick : stabilization term
    """
    max = np.amax(logp)
    ind_max = np.argmax(logp)
    logp = np.delete(logp, ind_max)
    trick = max + np.log(1 + np.sum(np.exp(logp - max)))

    return trick


def r_estimator(model, sm, H0, z):
    """
    Likelihood ratio estimator
    Inputs
        sm : (tensor)[nexamp x 1 x npix x npix] time delays mask
        H0 : (tensor)[nexamp x 1] Hubble constant
    Outputs
        lr : (array)[nexamp] likelihood ratio
    """

    sm = sm.to(device, non_blocking=True).float()
    H0 = H0.to(device, non_blocking=True).float()
    z = z.to(device, non_blocking=True).float()

    # model = model.to(device, non_blocking=True)
    model.eval()

    with torch.no_grad():
        output = model(sm, H0, z)

    sm.detach()
    H0.detach()
    z.detach()

    sm = torch.nn.Softmax(dim=1)
    prob = sm(output)
    s = prob[:, 1].detach().cpu().numpy()
    lr = s / (1 - s)

    return lr


def acc_fct(y_hat, y):
    """
    Computes the model's predictions accuracy
    Inputs
        y_hat : (tensor) [batch_size x 2] Model's prediction
        y : (tensor) [batch_size] Labels
    Outputs
        acc : (float) Accuracy
    """
    pred = torch.argmax(y_hat, axis=1)
    acc = torch.mean((y == pred).float())

    return acc


def split_data(file, path_in):
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
    dataset = h5py.File(os.path.join(path_in, file), 'r')
    dt = dataset["time_delays"][:]
    param = dataset["parameters"][:, :7]
    H0 = dataset["Hubble_cst"][:]
    z = dataset["redshifts"][:]
    pos = dataset["positions"][:]
    dataset.close()

    nsamp = H0.shape[0]
    dt = dt[dt != 0].reshape(nsamp, 3)

    # Splitting sets
    keys = np.arange(nsamp)
    tr_k = np.random.choice(keys, size=int(.8 * nsamp), replace=False)
    left_keys = np.setdiff1d(keys, tr_k)
    vl_k = np.random.choice(left_keys, size=int(.1 * nsamp), replace=False)
    ts_k = np.setdiff1d(left_keys, vl_k)

    dt_tr, H0_tr, z_tr, pr_tr, ps_tr = dt[tr_k], H0[tr_k], z[tr_k], param[tr_k], pos[tr_k]
    dt_vl, H0_vl, z_vl, pr_vl, ps_vl = dt[vl_k], H0[vl_k], z[vl_k], param[vl_k], pos[vl_k]

    # Saving keys
    if not os.path.isfile(path_in + '/keys.hdf5'):
        # os.remove(path_in+'/keys.hdf5')
        keys_file = h5py.File(path_in + '/keys.hdf5', 'a')
        train_ids = keys_file.create_dataset("train", tr_k.shape, dtype='i')
        valid_ids = keys_file.create_dataset("valid", vl_k.shape, dtype='i')
        test_ids = keys_file.create_dataset("test", ts_k.shape, dtype='i')
        train_ids[:] = tr_k
        valid_ids[:] = vl_k
        test_ids[:] = ts_k

    # Outputs
    train_set = [torch.from_numpy(dt_tr), torch.from_numpy(H0_tr), torch.from_numpy(z_tr),
                 torch.from_numpy(pr_tr), torch.from_numpy(ps_tr)]
    valid_set = [torch.from_numpy(dt_vl), torch.from_numpy(H0_vl), torch.from_numpy(z_vl),
                 torch.from_numpy(pr_vl), torch.from_numpy(ps_vl)]

    return train_set, valid_set


def train_fn(model, file, path_in, path_out, optimizer, loss_fn, acc_fn, threshold, sched=None,
             grad_clip=None, anomaly_detection=False, batch_size=256, epochs=100):
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
    train_set, valid_set = split_data(file, path_in)
    dt_tr, H0_tr, z_tr, pr_tr, ps_tr = train_set
    dt_vl, H0_vl, z_vl, pr_vl, ps_vl = valid_set

    dataset_train = torch.utils.data.TensorDataset(dt_tr, H0_tr, z_tr, pr_tr, ps_tr)
    dataset_valid = torch.utils.data.TensorDataset(dt_vl, H0_vl, z_vl, pr_vl, ps_vl)

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
            for dt, H0, z, param, pos in dataloader:

                dt = dt.to(device, non_blocking=True).float()
                H0 = H0.to(device, non_blocking=True).float()
                z = z.to(device, non_blocking=True).float()
                param = param.to(device, non_blocking=True).float()
                pos = pos.to(device, non_blocking=True).float()

                dt = dt_noise(dt)
                nim = torch.count_nonzero(dt + 1, dim=1)
                pot = doub_and_quad_potentials(nim, pos, param)
                sm = torch.cat((dt[:, :, None], pot[:, :, None]), dim=2)

                half_batch_size = int(dt.shape[0] / 2)

                sma = sm[:half_batch_size]
                smb = sm[half_batch_size:]

                H0a = H0[:half_batch_size]
                H0b = H0[half_batch_size:]

                za = z[:half_batch_size]
                zb = z[half_batch_size:]

                y_dep = torch.ones((half_batch_size)).to(device, non_blocking=True).long()
                y_ind = torch.zeros((half_batch_size)).to(device, non_blocking=True).long()

                # training phase
                if phase == 'train':
                    # Forward pass
                    for param in model.parameters():
                        param.grad = None

                    if anomaly_detection:
                        with autograd.detect_anomaly():
                            y_hat_a_dep = model(sma, H0a, za)
                            y_hat_a_ind = model(sma, H0b, za)
                            loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                            y_hat_b_dep = model(smb, H0b, zb)
                            y_hat_b_ind = model(smb, H0a, zb)
                            loss_b = loss_fn(y_hat_b_dep, y_dep) + loss_fn(y_hat_b_ind, y_ind)
                            loss = loss_a + loss_b
                    else:
                        y_hat_a_dep = model(sma, H0a, za)
                        y_hat_a_ind = model(sma, H0b, za)
                        loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                        y_hat_b_dep = model(smb, H0b, zb)
                        y_hat_b_ind = model(smb, H0a, zb)
                        loss_b = loss_fn(y_hat_b_dep, y_dep) + loss_fn(y_hat_b_ind, y_ind)
                        loss = loss_a + loss_b

                    # Backward Pass
                    loss.backward()

                    if grad_clip is not None:
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    optimizer.step()

                # validation phase
                else:
                    with torch.no_grad():
                        y_hat_a_dep = model(sma, H0a, za)
                        y_hat_a_ind = model(sma, H0b, za)
                        loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                        y_hat_b_dep = model(smb, H0b, zb)
                        y_hat_b_ind = model(smb, H0a, zb)
                        loss_b = loss_fn(y_hat_b_dep, y_dep) + loss_fn(y_hat_b_ind, y_ind)
                        loss = loss_a + loss_b

                # accuracy evaluation
                acc_a_dep = acc_fn(y_hat_a_dep, y_dep)
                acc_a_ind = acc_fn(y_hat_a_ind, y_ind)
                acc_b_dep = acc_fn(y_hat_b_dep, y_dep)
                acc_b_ind = acc_fn(y_hat_b_ind, y_ind)
                acc = (acc_a_dep + acc_a_ind + acc_b_dep + acc_b_ind) / 4

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
        if epoch % int(epochs / 25) == 0:
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
    plt.ylabel("Loss (Cross entropy)")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.legend()
    plt.savefig(path_out + '/loss.png', bbox_inches='tight', dpi=200)

    # accuracy
    plt.figure()
    plt.plot(100 * train_acc, 'b', label='Training')
    plt.plot(100 * valid_acc, 'r', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.legend()
    plt.savefig(path_out + '/acc.png', bbox_inches='tight', dpi=200)


def inference(file_keys, file_data, file_model, path_out, nrow=10, ncol=5, npts=5000, batch_size=200):
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
    keys = h5py.File(file_keys, 'r')
    test_keys = keys["test"][:]
    keys.close()

    # import data
    dataset = h5py.File(file_data, 'r')
    truths = dataset["Hubble_cst"][test_keys]
    lower_bound = np.floor(np.min(truths))
    higher_bound = np.ceil(np.max(truths))

    # remove out of bounds data
    idx_up = np.where(truths > 75.)[0]
    idx_down = np.where(truths < 65.)[0]
    idx_out = np.concatenate((idx_up, idx_down))
    truths = np.delete(truths, idx_out, axis=0)
    truths = truths.flatten()
    test_keys = np.delete(test_keys, idx_out, axis=0)
    dt = dataset["time_delays"][test_keys]
    z = dataset["redshifts"][test_keys]
    param = dataset["parameters"][test_keys, :7]
    pos = dataset["positions"][test_keys]
    dataset.close()

    nsamp = truths.shape[0]
    dt = dt[dt != 0].reshape(nsamp, 3)
    noisy_dt = dt_noise(torch.from_numpy(dt))
    nim = torch.count_nonzero(noisy_dt + 1, dim=1)
    pot = doub_and_quad_potentials(nim, torch.from_numpy(pos), torch.from_numpy(param))
    samples = torch.cat((noisy_dt[:, :, None], pot[:, :, None]), dim=2)

    # observations
    samples_rep = torch.repeat_interleave(samples, npts, dim=0)
    z_rep = torch.repeat_interleave(torch.from_numpy(z), npts, dim=0)

    # Global NRE posterior
    prior = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    prior_tile = np.tile(prior, (nsamp, 1))
    prior = prior.flatten()

    dataset_test = torch.utils.data.TensorDataset(samples_rep, torch.from_numpy(prior_tile), z_rep)
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    post = []
    for samp, H0, shift in dataloader:
        ratios = r_estimator(model, samp, H0, shift)
        post.extend(ratios)

    post = np.asarray(post).reshape(nsamp, npts)
    prior_tile = prior_tile.reshape(nsamp, npts)
    post = normalization(prior_tile, post)

    # predictions
    arg_pred = np.argmax(post, axis=1)
    pred = prior_tile[np.arange(nsamp), arg_pred]

    it = 0
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=True, figsize=(3 * ncol, 3 * nrow))
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].plot(prior, post[it], '-', label='{:.2f}'.format(pred[it]))
            min_post = np.min(post[it])
            max_post = np.max(post[it])
            axes[i, j].vlines(truths[it], min_post, max_post, colors='black', linestyles='dotted',
                              label='{:.2f}'.format(truths[it]))
            axes[i, j].legend(frameon=False, borderpad=.2, handlelength=.6, fontsize=9, handletextpad=.4)
            if np.count_nonzero(dt[it] + 1) == 3:
                axes[i, j].set_title("Quad")
            if np.count_nonzero(dt[it] + 1) == 1:
                axes[i, j].set_title("Double")
            if i == int(nrow - 1):
                axes[i, j].set_xlabel(r"H$_0$ (km Mpc$^{-1}$ s$^{-1}$)")
            if j == 0:
                axes[i, j].set_ylabel(r"$p(H_0 \mid \Delta t, \Delta \phi)$")
            it += 1

    # saving
    plt.savefig(path_out + '/posteriors.png', bbox_inches='tight')

    # file to save
    if os.path.isfile(path_out + "/posteriors.hdf5"):
        os.remove(path_out + "/posteriors.hdf5")
    post_file = h5py.File(path_out + "/posteriors.hdf5", 'a')

    NRE = post_file.create_group("NRE_global")
    H0 = NRE.create_dataset("Hubble_cst", (npts,), dtype='f')
    pstr = NRE.create_dataset("posterior", (nsamp, npts), dtype='f')

    truth_set = post_file.create_dataset("truth", (nsamp,), dtype='f')

    H0[:] = prior
    pstr[:, :] = post
    truth_set[:] = truths

    post_file.close()

    # Highest
    credibility = np.zeros((nsamp,))
    for i in range(nsamp):

        idx_truth = np.where(abs(prior - truths[i]) == np.min(abs(prior - truths[i])))[0]
        probs = post[i]
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

        credibility[i] = np.trapz(probs[idx_HPD], prior[idx_HPD])

    # Coverage diagnostic
    bins = np.linspace(0., 1., 100)
    emperical_coverage = np.zeros_like(bins)
    for i in range(len(bins)):
        emperical_coverage[i] = np.mean(np.where(credibility <= bins[i], 1, 0))

    plt.figure()
    plt.plot(bins, bins, '--k')
    plt.plot(bins, emperical_coverage, '-', color="darkorange")
    plt.xlabel("Highest probability density region")
    plt.ylabel("Fraction of truths within")
    plt.text(0., .9, "Underconfident")  # , fontsize='large')
    plt.text(.6, .05, "Overconfident")  # , fontsize='large')
    plt.savefig(path_out + '/coverage.pdf', bbox_inches='tight', dpi=200)


def joint_inference(file_data, file_model, path_out, npts=50000, lower_bound=65., higher_bound=75., batch_size=200):
    """
        Performs the joint inference on a population of lenses
        Inputs
            file_data : (str) name of the file containing data
            file_model : (str) name of the file containing the model
            path_out : (str) directory where to save the output
        Outputs
            None
    """
    if device.type == "cpu":
        model = torch.load(file_model, map_location='cpu')
    else:
        model = torch.load(file_model)

    # import data
    dataset = h5py.File(file_data, 'r')
    truths = dataset["Hubble_cst"][:]
    dt = dataset["time_delays"][:]
    z = dataset["redshifts"][:]
    param = dataset["parameters"][:, :7]
    pos = dataset["positions"][:]
    dataset.close()

    true = float(np.unique(truths))
    nsamp = truths.shape[0]
    dt = dt[dt != 0].reshape(nsamp, 3)
    noisy_dt = dt_noise(torch.from_numpy(dt))
    nim = torch.count_nonzero(noisy_dt + 1, dim=1)
    pot = doub_and_quad_potentials(nim, torch.from_numpy(pos), torch.from_numpy(param))
    samples = torch.cat((noisy_dt[:, :, None], pot[:, :, None]), dim=2)

    # observations
    samples_rep = torch.repeat_interleave(samples, npts, dim=0)
    z_rep = torch.repeat_interleave(torch.from_numpy(z), npts, dim=0)

    # Global NRE posterior
    prior = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    prior_tile = np.tile(prior, (nsamp, 1))
    prior = prior.flatten()

    dataset_test = torch.utils.data.TensorDataset(samples_rep, torch.from_numpy(prior_tile), z_rep)
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    post = []
    for samp, H0, shift in dataloader:
        preds = r_estimator(model, samp, H0, shift)
        post.extend(preds)

    post = np.asarray(post).reshape(nsamp, npts)
    prior_tile = prior_tile.reshape(nsamp, npts)
    post = normalization(prior_tile, post)

    joint1 = np.sum(np.log(post[:50]), axis=0)
    joint1 -= log_trick(joint1)
    joint2 = np.sum(np.log(post[:500]), axis=0)
    joint2 -= log_trick(joint2)
    joint3 = np.sum(np.log(post[:3000]), axis=0)
    joint3 -= log_trick(joint3)
    joint4 = np.sum(np.log(post[:8000]), axis=0)
    joint4 -= log_trick(joint4)
    joints = np.concatenate((joint1[:, None], joint2[:, None], joint3[:, None], joint4[:, None]), axis=1)

    plt.figure()
    plt.plot(prior, np.exp(joint4), '-', zorder=0, label='8,000')
    plt.plot(prior, np.exp(joint3), '--', zorder=5, label='3,000')
    plt.plot(prior, np.exp(joint2), '-.', zorder=10, label='500')
    plt.plot(prior, np.exp(joint1), ':', zorder=15, label='50')
    min_post = np.min(np.exp(joints))
    max_post = np.max(np.exp(joints))
    plt.vlines(true, min_post, max_post, color='black', linestyles='solid', zorder=20, label="True value")
    plt.legend()
    plt.xlim([69.7, 70.3])
    plt.xlabel(r"H$_0$ (km Mpc$^{-1}$ s$^{-1}$)")
    plt.ylabel(r"$p(H_0 \mid \Delta t, \Delta \phi)$")
    plt.savefig(path_out + '/inference.pdf', bbox_inches='tight', dpi=200)

    plt.figure()
    plt.plot(prior, np.exp(joint4), '-', label='8,000')
    min_post = np.min(np.exp(joint4))
    max_post = np.max(np.exp(joint4))
    plt.vlines(true, min_post, max_post, color='black', linestyles='solid', label="True value")
    plt.legend()
    plt.xlim([69.9, 70.1])
    plt.xlabel(r"H$_0$ (km Mpc$^{-1}$ s$^{-1}$)")
    plt.ylabel(r"$p(H_0 \mid \Delta t, \Delta \phi)$")
    plt.savefig(path_out + '/test_bias.pdf', bbox_inches='tight', dpi=200)