# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
from torch.nn.utils import clip_grad_norm_
from torch import autograd

from lens_modeling import ellip_coordinates, shear_coordinates, ellip_polar, shear_polar, get_Fermat_potentials
from functions import r_estimator, normalization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)
np.random.seed(1)


# --- Functions for training -----------------------------------------------


def gaussian_noise(x, sig_dt=.41):  ###
    """
    Adds noise to time delays
    Inputs
        x : (tensor)[batch_size x 4 x 2] Time delays and Fermat potentials
        sig_dt : (float) noise standard deviation on time delays
        sig_pot : (float) noise standard deviation on potentials
    Outputs
        noisy_data : (tensor)[batch_size x 4 x 2] noisy time delays + true Fermat potential
    """
    mask_pad = torch.where(x[:, :, 0] == -1, 0, 1)
    mask_zero = torch.where(x[:, :, 0] == 0, 0, 1)  ###
    noise_dt = sig_dt * torch.randn((x.size(0), x.size(1)), device="cpu")
    noisy_data = x.clone()
    noisy_data[:, :, 0] += noise_dt * mask_pad * mask_zero  ###

    return noisy_data


def modeling(images, file_cnn, std_noise=.1, batch_size=128):
    """
    Inputs
        file_keys : (str) name of the file containing the keys of the test set
        file_data : (str) name of the file containing data
        file_cnn : (str) name of the file containing the model
        path_out : (str) directory where to save the output
    Outputs
        None
    """
    if torch.cuda.is_available():
        model = torch.load(file_cnn)
    else:
        model = torch.load(file_cnn, map_location="cpu")

    noisy_images = torch.from_numpy(images).to("cpu") + std_noise * torch.randn(images.shape, device="cpu")
    dataset_test = torch.utils.data.TensorDataset(noisy_images)
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    preds = []
    model = model.to(device, non_blocking=True)
    model.eval()
    for x in dataloader:
        x = x[0].to(device, non_blocking=True).float()
        with torch.no_grad():
            y_hat = model(x)
        preds.extend(y_hat.detach().cpu().numpy())

    preds = np.asarray(preds)

    return preds


def split_data(path_data, file_cnn):
    """
    Reads data, generates classes and splits data into training, validation and test sets
    Inputs
        file : (str) name of the file containing data
        path_data : (str) file's directory
    Outputs
        All outputs are a list of the time delays, the Fermat potential, the H0 and the labels tensors
        train_set : (list) training set. Corresponds to 80% of all data.
        valid_set : (list) validation set. Corresponds to 10% of all data.
    """
    # Reading data
    dataset = h5py.File(os.path.join(path_data, "dataset.hdf5"), 'r')
    dt = torch.from_numpy(dataset["time_delays"][:])
    H0 = torch.from_numpy(dataset["Hubble_cst"][:])
    z = torch.from_numpy(dataset["redshifts"][:])
    images = dataset["hosts"][:]
    im_pos = torch.from_numpy(dataset["positions"][:])
    param = dataset["parameters"][:, :7]
    dataset.close()

    images = images / np.amax(images, axis=(1, 2, 3), keepdims=True)
    param[:, 2], param[:, 3] = ellip_coordinates(param[:, 2], param[:, 3])
    param[:, 5], param[:, 6] = shear_coordinates(param[:, 5], param[:, 6])
    mean = np.mean(param, axis=0)
    std = np.std(param, axis=0)
    param_pred = modeling(images, file_cnn) * std + mean
    param_pred[:, 2], param_pred[:, 3] = ellip_polar(param_pred[:, 2], param_pred[:, 3])
    param_pred[:, 5], param_pred[:, 6] = shear_polar(param_pred[:, 5], param_pred[:, 6])
    param_pred = torch.from_numpy(param_pred)

    nsamp = dt.shape[0]
    nim = torch.count_nonzero(dt + 1, dim=1)

    ind2 = torch.where(nim == 2)
    pos_doub = im_pos[ind2][:, :, :-2]
    param_doub = param_pred[ind2]
    pot_doub = get_Fermat_potentials(param_doub[:, 0], param_doub[:, 1], param_doub[:, 4],
                                     param_doub[:, 2], param_doub[:, 3], param_doub[:, 5],
                                     param_doub[:, 6], pos_doub[:, 0], pos_doub[:, 1])
    pot_doub = torch.cat((pot_doub, -torch.ones((pot_doub.shape[0], 2), device=pot_doub.device)), dim=1)

    ind4 = torch.where(nim == 4)
    pos_quad = im_pos[ind4]
    param_quad = param_pred[ind4]
    pot_quad = get_Fermat_potentials(param_quad[:, 0], param_quad[:, 1], param_quad[:, 4],
                                     param_quad[:, 2], param_quad[:, 3], param_quad[:, 5],
                                     param_quad[:, 6], pos_quad[:, 0], pos_quad[:, 1])

    pot = torch.ones((nsamp, 4), device=pot_doub.device)
    pot[ind2] = pot_doub
    pot[ind4] = pot_quad

    samples = torch.cat((dt[:, :, None], pot[:, :, None]), dim=2)
    samples = samples[samples != 0].reshape(nsamp, 3, 2)

    # import keys
    keys = h5py.File(os.path.join(path_data, "keys.hdf5"), 'r')
    train_keys = keys["train"][:]
    valid_keys = keys["valid"][:]
    keys.close()

    x1_train, x2_train, x3_train = samples[train_keys], H0[train_keys], z[train_keys]
    x1_valid, x2_valid, x3_valid = samples[valid_keys], H0[valid_keys], z[valid_keys]

    # Outputs
    train_set = [x1_train, x2_train, x3_train]
    valid_set = [x1_valid, x2_valid, x3_valid]

    return train_set, valid_set


def train_fn(model, file_cnn, path_data, path_out, optimizer, loss_fn, acc_fn, threshold, sched=None,
             grad_clip=None, anomaly_detection=False, batch_size=256, epochs=100):
    """
    Manages the training
    Inputs
        model : (module object) neural network
        file : (str) name of the file containing data
        path_data : (str) path to data
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
    train_set, valid_set = split_data(path_data, file_cnn)
    x1_train, x2_train, x3_train = train_set
    x1_val, x2_val, x3_val = valid_set

    dataset_train = torch.utils.data.TensorDataset(x1_train, x2_train, x3_train)
    dataset_valid = torch.utils.data.TensorDataset(x1_val, x2_val, x3_val)

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
            for x1, x2, x3 in dataloader:
                half_batch_size = int(x1.shape[0] / 2)

                x1 = gaussian_noise(x1)
                x1 = x1.to(device, non_blocking=True).float()
                x1a = x1[:half_batch_size]
                x1b = x1[half_batch_size:]

                x2 = x2.to(device, non_blocking=True).float()
                x2a = x2[:half_batch_size]
                x2b = x2[half_batch_size:]

                x3 = x3.to(device, non_blocking=True).float()
                x3a = x3[:half_batch_size]
                x3b = x3[half_batch_size:]

                y_dep = torch.ones((half_batch_size)).to(device, non_blocking=True).long()
                y_ind = torch.zeros((half_batch_size)).to(device, non_blocking=True).long()

                # training phase
                if phase == 'train':
                    # Forward pass
                    for param in model.parameters():
                        param.grad = None

                    if anomaly_detection:
                        with autograd.detect_anomaly():
                            y_hat_a_dep = model(x1a, x2a, x3a)
                            y_hat_a_ind = model(x1a, x2b, x3a)
                            loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                            y_hat_b_dep = model(x1b, x2b, x3b)
                            y_hat_b_ind = model(x1b, x2a, x3b)
                            loss_b = loss_fn(y_hat_b_dep, y_dep) + loss_fn(y_hat_b_ind, y_ind)
                            loss = loss_a + loss_b
                    else:
                        y_hat_a_dep = model(x1a, x2a, x3a)
                        y_hat_a_ind = model(x1a, x2b, x3a)
                        loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                        y_hat_b_dep = model(x1b, x2b, x3b)
                        y_hat_b_ind = model(x1b, x2a, x3b)
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
                        y_hat_a_dep = model(x1a, x2a, x3a)
                        y_hat_a_ind = model(x1a, x2b, x3a)
                        loss_a = loss_fn(y_hat_a_dep, y_dep) + loss_fn(y_hat_a_ind, y_ind)
                        y_hat_b_dep = model(x1b, x2b, x3b)
                        y_hat_b_ind = model(x1b, x2a, x3b)
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


def inference(path_data, file_model, file_cnn, path_out, nrow=5, ncol=4, npts=1000, batch_size=100):
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
    z = dataset["redshifts"][test_keys]
    param = dataset["parameters"][:, :7]
    images = dataset["hosts"][:]
    im_pos = torch.from_numpy(dataset["positions"][test_keys])
    dataset.close()

    images = images / np.amax(images, axis=(1, 2, 3), keepdims=True)
    param[:, 2], param[:, 3] = ellip_coordinates(param[:, 2], param[:, 3])
    param[:, 5], param[:, 6] = shear_coordinates(param[:, 5], param[:, 6])
    mean = np.mean(param, axis=0)
    std = np.std(param, axis=0)
    param_pred = modeling(images, file_cnn) * std + mean

    param_pred[:, 2], param_pred[:, 3] = ellip_polar(param_pred[:, 2], param_pred[:, 3])
    param_pred[:, 5], param_pred[:, 6] = shear_polar(param_pred[:, 5], param_pred[:, 6])
    param_pred = torch.from_numpy(np.delete(param_pred, idx_out, axis=0))

    nsamp = time_delays.shape[0]
    dt = torch.from_numpy(time_delays)
    nim = torch.count_nonzero(dt + 1, dim=1)

    ind2 = torch.where(nim == 2)
    pos_doub = im_pos[ind2][:, :, :-2]
    param_doub = param_pred[ind2]
    pot_doub = get_Fermat_potentials(param_doub[:, 0], param_doub[:, 1], param_doub[:, 4],
                                     param_doub[:, 2], param_doub[:, 3], param_doub[:, 5],
                                     param_doub[:, 6], pos_doub[:, 0], pos_doub[:, 1])
    pot_doub = torch.cat((pot_doub, -torch.ones((pot_doub.shape[0], 2), device=pot_doub.device)), dim=1)

    ind4 = torch.where(nim == 4)
    pos_quad = im_pos[ind4]
    param_quad = param_pred[ind4]
    pot_quad = get_Fermat_potentials(param_quad[:, 0], param_quad[:, 1], param_quad[:, 4],
                                     param_quad[:, 2], param_quad[:, 3], param_quad[:, 5],
                                     param_quad[:, 6], pos_quad[:, 0], pos_quad[:, 1])

    pot = torch.ones((nsamp, 4), device=pot_doub.device)
    pot[ind2] = pot_doub
    pot[ind4] = pot_quad

    data = torch.cat((dt[:, :, None], pot[:, :, None]), dim=2)
    data = gaussian_noise(data)
    data = data[data != 0].reshape(nsamp, 3, 2)
    data_repeated = torch.repeat_interleave(data, npts, dim=0)
    z_repeated = torch.repeat_interleave(torch.from_numpy(z), npts, dim=0)

    # Global NRE posterior
    support = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    support_tile = np.tile(support, (nsamp, 1))
    support = support.flatten()

    dataset_test = torch.utils.data.TensorDataset(data_repeated, torch.from_numpy(support_tile), z_repeated)
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    ratios = []
    for x1, x2, x3 in dataloader:
        preds = r_estimator(model, x1, x2, x3)
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