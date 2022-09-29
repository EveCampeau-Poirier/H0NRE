# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
c = c.to('Mpc/d')  # Speed of light

# PyTorch libraries
import torch
from torch.nn.utils import clip_grad_norm_
from torch import autograd
from torchquad import Simpson, set_up_backend
if torch.cuda.is_available():
    set_up_backend("cuda:0")

torch.manual_seed(0)
np.random.seed(0)


# --- Functions for training -----------------------------------------------

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


def gaussian_noise(x, sig_dt=.3, sig_pot=.003):
    """
    Adds noise to time delays
    Inputs
        x : (tensor)[batch_size x 4 x 2] Time delays and Fermat potentials
        sig_dt : (float) noise standard deviation on time delays
        sig_pot : (float) noise standard deviation on potentials
    Outputs
        noisy_data : (tensor)[batch_size x 4 x 2] noisy time delays + true Fermat potential
    """
    mask = torch.where(x[:, :, 0] == -1, 0, 1)
    noise_dt = sig_dt * torch.randn((x.size(0), x.size(1)))
    noise_pot = sig_pot * torch.randn((x.size(0), x.size(1)))
    noisy_data = x.clone()
    noisy_data[:, :, 0] += noise_dt * mask
    noisy_data[:, :, 1] += noise_pot * mask

    return noisy_data


def gaussian(x, mu, sigma):
    """
        Computes on multivariate gaussian with a diagonal covariance matrix
        Inputs
            x : (array) variable
            mu : (array) mean
            sigma : (float) standard deviation
        Outputs
            gauss : (float) Gaussian's value at x
        """
    gauss = torch.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    return gauss


def time_delays_eq(fermat, H0, zs, zd):
    """
    Function to integrate to estimate the posterior distribution
    Inputs
        fermat : (float) Potential value
        H0 : (float) H0 value
        zs : (float) Source's redshift
        zd : (float) Deflector's redshift
    Outputs
        time_delays : (float) Time delay generated by fermat
    """
    cosmo_model = FlatLambdaCDM(H0=H0, Om0=.3)
    Ds = cosmo_model.angular_diameter_distance(zs)
    Dd = cosmo_model.angular_diameter_distance(zd)
    Dds = cosmo_model.angular_diameter_distance_z1z2(zd, zs)
    time_delays = (1 + zd) * Dd.value * Ds.value / Dds.value / c.value * fermat
    time_delays *= (2 * np.pi / 360 / 3600) ** 2  # Conversion to days

    return time_delays


def analytical_posterior(time_delays, fermat_pot, H0, zs, zd, sig_dt=.3, sig_pot=.003):
    """
    Computes the analytical posterior
    Inputs
        time_delays : (array) [nsamp x 3] Measured time delays
        fermat_pot : (array) [nsamp x 3] Modeled Fermat potentials
        H0 : (array) [npts_post] Grid of H0 values
        zs : (array) Source's redshift
        zd : (array) Deflector's redshift
        sig_dt : (float) Time delays standard deviation
        sig_pot : (float) Fermat potentils standard deviation
    Outputs
        post : (array)[nsamp x npts] posterior
    """

    nsamp = time_delays.shape[0]
    npts = H0.shape[0]
    post = torch.ones((nsamp, npts)).to(device)
    simp = Simpson()

    for i in range(nsamp):
        size = np.count_nonzero(time_delays[i] + 1.)
        dt = time_delays[i][time_delays[i] != -1].reshape(size)
        pot = fermat_pot[i][fermat_pot[i] != -1].reshape(size)

        dt = torch.from_numpy(dt).to(device)
        pot = torch.from_numpy(pot).to(device)

        if size == 1:
            limit_dt = [dt - 3 * sig_dt, dt + 3 * sig_dt]
            limit_dphi = [pot - 3 * sig_pot, pot + 3 * sig_pot]

        if size == 3:
            limit_dt1 = [dt[0] - 3 * sig_dt, dt[0] + 3 * sig_dt]
            limit_dt2 = [dt[1] - 3 * sig_dt, dt[1] + 3 * sig_dt]
            limit_dt3 = [dt[2] - 3 * sig_dt, dt[2] + 3 * sig_dt]
            limit_dphi1 = [pot[0] - 3 * sig_pot, pot[0] + 3 * sig_pot]
            limit_dphi2 = [pot[1] - 3 * sig_pot, pot[1] + 3 * sig_pot]
            limit_dphi3 = [pot[2] - 3 * sig_pot, pot[2] + 3 * sig_pot]

        for j in range(npts):

            if size == 1:
                def integrand(x):
                    """
                    Function to integrate to estimate the posterior distribution
                    Inputs
                        x : sample
                    Outputs
                        I : (float) Evaluation of the integrand
                    """
                    I = gaussian(x[:, 0], time_delays_eq(x[:, 1], H0[j], zs[i], zd[i]), sig_dt) * \
                        gaussian(x[:, 0], dt, sig_dt) * gaussian(x[:, 1], pot, sig_pot)

                    return I

                I = simp.integrate(integrand, dim=2, N=10000,
                                   integration_domain=[limit_dt, limit_dphi])
                post[i, j] = I

            if size == 3:
                def integrand(x):
                    """
                    Function to integrate to estimate the posterior distribution
                    Inputs
                        x : samples
                    Outputs
                        I : (float) Evaluation of the integrand
                    """

                    I1 = gaussian(x[:, 0], time_delays_eq(x[:, 1], H0[j], zs[i], zd[i]), sig_dt) * \
                         gaussian(x[:, 0], dt[0], sig_dt) * gaussian(x[:, 1], pot[0], sig_pot)
                    I2 = gaussian(x[:, 2], time_delays_eq(x[:, 3], H0[j], zs[i], zd[i]), sig_dt) * \
                         gaussian(x[:, 2], dt[1], sig_dt) * gaussian(x[:, 3], pot[1], sig_pot)
                    I3 = gaussian(x[:, 4], time_delays_eq(x[:, 5], H0[j], zs[i], zd[i]), sig_dt) * \
                         gaussian(x[:, 4], dt[2], sig_dt) * gaussian(x[:, 5], pot[2], sig_pot)
                    I = I1 * I2 * I3

                    return I

                I = simp.integrate(integrand, dim=6, N=10000,
                                   integration_domain=[limit_dt1, limit_dphi1, limit_dt2, limit_dphi2, limit_dt3,
                                                       limit_dphi3])
                post[i, j] = I

    return post.detach().cpu().numpy()


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
    pot = dataset["Fermat_potential"][:]
    H0 = dataset["Hubble_cst"][:]
    dataset.close()

    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)
    nsamp = samples.shape[0]
    samples = samples[samples != 0]
    samples = samples.reshape(nsamp, 3, 2)

    # Splitting sets
    keys = np.arange(nsamp)
    train_keys = np.random.choice(keys, size=int(.8 * nsamp), replace=False)
    left_keys = np.setdiff1d(keys, train_keys)
    valid_keys = np.random.choice(left_keys, size=int(.1 * nsamp), replace=False)
    test_keys = np.setdiff1d(left_keys, valid_keys)

    x1_train, x2_train = samples[train_keys], H0[train_keys]
    x1_valid, x2_valid = samples[valid_keys], H0[valid_keys]

    # Saving keys
    if not os.path.isfile(path_in + '/keys.hdf5'):
        # os.remove(path_in+'/keys.hdf5')
        keys_file = h5py.File(path_in + '/keys.hdf5', 'a')
        train_ids = keys_file.create_dataset("train", train_keys.shape, dtype='i')
        valid_ids = keys_file.create_dataset("valid", valid_keys.shape, dtype='i')
        test_ids = keys_file.create_dataset("test", test_keys.shape, dtype='i')
        train_ids[:] = train_keys
        valid_ids[:] = valid_keys
        test_ids[:] = test_keys

    # Outputs
    train_set = [torch.from_numpy(x1_train), torch.from_numpy(x2_train)]
    valid_set = [torch.from_numpy(x1_valid), torch.from_numpy(x2_valid)]

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model and loss function on GPU
    model = model.to(device, non_blocking=True)
    loss_fn = loss_fn.to(device, non_blocking=True)

    # Datasets
    train_set, valid_set = split_data(file, path_in)
    x1_train, x2_train = train_set
    x1_val, x2_val = valid_set

    dataset_train = torch.utils.data.TensorDataset(x1_train, x2_train)
    dataset_valid = torch.utils.data.TensorDataset(x1_val, x2_val)

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
            for x1, x2 in dataloader:
                half_batch_size = int(x1.shape[0] / 2)

                x1 = gaussian_noise(x1)
                x1 = x1.to(device, non_blocking=True).float()
                x1a = x1[:half_batch_size]
                x1b = x1[half_batch_size:]

                x2 = x2.to(device, non_blocking=True).float()
                x2a = x2[:half_batch_size]
                x2b = x2[half_batch_size:]

                y_dep = torch.ones((half_batch_size)).to(device, non_blocking=True).long()
                y_ind = torch.zeros((half_batch_size)).to(device, non_blocking=True).long()

                # training phase
                if phase == 'train':
                    # Forward pass
                    for param in model.parameters():
                        param.grad = None

                    if anomaly_detection:
                        with autograd.detect_anomaly():
                            y_hat_a_dep = model(x1a, x2a)
                            y_hat_a_ind = model(x1a, x2b)
                            loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                            y_hat_b_dep = model(x1b, x2b)
                            y_hat_b_ind = model(x1b, x2a)
                            loss_b = loss_fn(y_hat_b_dep, y_dep) + loss_fn(y_hat_b_ind, y_ind)
                            loss = loss_a + loss_b
                    else:
                        y_hat_a_dep = model(x1a, x2a)
                        y_hat_a_ind = model(x1a, x2b)
                        loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                        y_hat_b_dep = model(x1b, x2b)
                        y_hat_b_ind = model(x1b, x2a)
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
                        y_hat_a_dep = model(x1a, x2a)
                        y_hat_a_ind = model(x1a, x2b)
                        loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                        y_hat_b_dep = model(x1b, x2b)
                        y_hat_b_ind = model(x1b, x2a)
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
    plt.ylabel("Loss (Cross entropy)")
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


def inference(file_keys, file_data, file_model, path_out, nrow=5, ncol=4, npts=1000, batch_size=100):
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
    pot = dataset["Fermat_potential"][test_keys]
    z = dataset["redshifts"][test_keys]
    dataset.close()

    # reshape data
    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)
    samples = samples[:4000]
    nsamp = samples.shape[0]
    samples = samples[samples != 0]
    samples = samples.reshape(nsamp, 3, 2)

    # observations
    noisy_data = gaussian_noise(torch.from_numpy(samples))
    noisy_data_repeated = torch.repeat_interleave(x, npts, dim=0)

    # Global NRE posterior
    support = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    support_tile = np.tile(support, (nsamp, 1))

    dataset_test = torch.utils.data.TensorDataset(noisy_data_repeated, torch.from_numpy(support_tile))
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

    # analytical posterior
    analytic = analytical_posterior(noisy_data[:, :, 0].numpy(), noisy_data[:, :, 1].numpy(), support, z[:, 1], z[:, 0])
    analytic = normalization(support, analytic)

    it = 0
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=False, figsize=(3 * ncol, 3 * nrow))
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].plot(support, analytic[it], '--g', label='{:.2f}'.format(support[np.argmax(analytic[it])]))
            axes[i, j].plot(support, ratios[it], '-b', label='{:.2f}'.format(pred[it]))
            min_post = np.minimum(np.min(ratios[it]), np.min(analytic[it]))
            max_post = np.maximum(np.max(ratios[it]), np.max(analytic[it]))
            axes[i, j].vlines(truths[it], min_post, max_post, colors='r', linestyles='dotted',
                              label='{:.2f}'.format(truths[it]))
            axes[i, j].legend(frameon=False, borderpad=.2, handlelength=.6, fontsize=9, handletextpad=.4)
            if np.count_nonzero(samples[it, :, 0] + 1) == 3:
                axes[i, j].set_title("Quad")
            if np.count_nonzero(samples[it, :, 0] + 1) == 1:
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
    post_anl = NRE.create_dataset("analytical", (nsamp, npts), dtype='f')

    truth_set = post_file.create_dataset("truth", (nsamp,), dtype='f')

    H0[:] = support.flatten()
    post[:, :] = ratios
    post_anl[:, :] = analytic
    truth_set[:] = truths[:nsamp]

    post_file.close()

    # integration from pred to true
    credibility = np.zeros((nsamp,))
    for i in range(nsamp):

        idx_truth = np.where(abs(support - truths[i]) == np.min(abs(support - truths[i])))[0]
        probs = ratios[i]
        prob_truth = probs[idx_truth]

        prob_false = probs.copy()
        prob_false[idx_truth] = 0.
        idx_equal_prob = np.where(abs(prob_false - prob_truth) == np.min(abs(prob_false - prob_truth)))

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


def joint_inference(file_data, file_model, path_out, npts=1000, lower_bound=60., upper_bound=80.):
    """
        Performs the joint inference on a population of lenses
        Inputs
            file_data : (str) name of the file containing data
            file_model : (str) name of the file containing the model
            path_out : (str) directory where to save the output
        Outputs
            None
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == "cpu":
        model = torch.load(file_model, map_location='cpu')
    else:
        model = torch.load(file_model)

    # import data
    dataset = h5py.File(file_data, 'r')
    H0 = dataset["Hubble_cst"][:]
    dt = dataset["time_delays"][:]
    pot = dataset["Fermat_potential"][:]
    dataset.close()
    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)
    samples = samples[:1000]
    nsamp = samples.shape[0]
    samples = samples[samples != 0]
    samples = samples.reshape(nsamp, 3, 2)
    true = float(np.unique(H0))

    # observations
    x = gaussian_noise(torch.from_numpy(samples))
    data = torch.repeat_interleave(x, npts, dim=0)

    # Global NRE posterior
    prior = np.linspace(lower_bound, upper_bound, npts).reshape(npts, 1)
    prior_tile = np.tile(prior, (nsamp, 1))
    prior = prior.flatten()
    post = r_estimator(model, data, torch.from_numpy(prior_tile))

    post = post.reshape(nsamp, npts)
    post = normalization(prior_tile.reshape(nsamp, npts), post)

    joint1 = np.sum(np.log(post[:10]), axis=0)
    joint1 -= log_trick(joint1)
    joint2 = np.sum(np.log(post[:100]), axis=0)
    joint2 -= log_trick(joint2)
    joint3 = np.sum(np.log(post[:1000]), axis=0)
    joint3 -= log_trick(joint3)
    joints = np.concatenate((joint1[:, None], joint2[:, None], joint3[:, None]), axis=1)

    plt.figure()
    plt.plot(prior, np.exp(joint1), '-b', label='10 lenses')
    plt.plot(prior, np.exp(joint2), '--r', label='100 lenses')
    plt.plot(prior, np.exp(joint3), '-.g', label='1000 lenses')
    min_post = np.min(np.exp(joints))
    max_post = np.max(np.exp(joints))
    plt.vlines(true, min_post, max_post, colors='k', linestyles='dotted', label="truth")
    plt.legend()
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.savefig(path_out + '/inference.png', bbox_inches='tight')