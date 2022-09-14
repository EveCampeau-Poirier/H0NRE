import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import argparse
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functions import normalization

from astropy.cosmology import FlatLambdaCDM

from simulator import training_set

ts = training_set()

# Allow reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def analytical_likelihood(dt, pot, H0, zs, zd, sig_dt=.3, sig_pot=.003):
    """
    Computes the analytical likelihood
    Inputs
        x : (array)[test set size x 4] Noisy time delays
        mu : (array)[test set size x 4] True time delays
        sigma : (float) noise standard deviation on time delays
    Outputs
        lkh_doub : (tensor)[nsamp x npts] Likelihood
    """
    nsamp = dt.shape[0]
    npts = H0.shape[0]
    mu = np.zeros((nsamp, npts, 4))
    pad = -np.ones((2))

    for i in range(nsamp):
        for j in range(npts):
            fermat = pot[i]
            fermat = fermat[fermat != -1]
            cosmo_model = FlatLambdaCDM(H0=H0[j], Om0=.3)
            Ds = cosmo_model.angular_diameter_distance(zs[i])
            Dd = cosmo_model.angular_diameter_distance(zd[i])
            Dds = cosmo_model.angular_diameter_distance_z1z2(zd[i], zs[i])
            sim = ts.get_time_delays([zs[i], zd[i], Ds.value, Dd.value, Dds.value, 0, H0[j]], [0, 0, 0, fermat])
            if len(fermat) == 2:
                sim = np.concatenate((sim, pad), axis=None)
            mu[i, j] = sim

    size = np.count_nonzero(dt + 1, axis=1)
    lkh = np.exp(-np.sum((dt[:, None] - mu) ** 2, axis=2) / 2 / sig_dt ** 2) / (2 * np.pi * sig_dt ** 2) ** size[:,
                                                                                                            None]

    return lkh


def get_labels(x2):
    nsamp = x2.shape[0]
    idx_dep = np.arange(int(nsamp / 2))
    idx_ind = np.arange(int(nsamp / 2), nsamp)
    random.shuffle(idx_ind)
    idx = np.concatenate((idx_dep, idx_ind), axis=0)
    labels = np.concatenate((np.ones(idx_dep.shape[0]), np.zeros(idx_ind.shape[0])), axis=0)
    return x2[idx], labels


def get_datasets(file_keys, file_data):
    # import keys
    keys = h5py.File(file_keys, 'r')
    calib_keys = keys["test"][:]
    keys.close()

    nsamp = calib_keys.shape[0]
    keys = np.arange(nsamp)
    train_keys = np.random.choice(keys, size=int(.8 * nsamp), replace=False)
    keys = np.setdiff1d(keys, train_keys)
    valid_keys = np.random.choice(keys, size=int(.1 * nsamp), replace=False)
    test_keys = np.setdiff1d(keys, valid_keys)

    # Reading data
    dataset = h5py.File(file_data, 'r')
    dt = dataset["time_delays"][calib_keys]
    pot = dataset["Fermat_potential"][calib_keys]
    H0 = dataset["Hubble_cst"][calib_keys]
    dataset.close()
    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)
    # samples = samples[samples != 0]
    # samples = samples.reshape(nsamp, 3, 2)

    x1_train, x2_train = samples[train_keys], H0[train_keys]
    x1_valid, x2_valid = samples[valid_keys], H0[valid_keys]
    x1_test, x2_test = samples[test_keys], H0[test_keys]

    x2_train, y_train = get_labels(x2_train)
    x2_valid, y_valid = get_labels(x2_valid)

    train_set = [torch.from_numpy(x1_train), torch.from_numpy(x2_train), torch.from_numpy(y_train)]
    valid_set = [torch.from_numpy(x1_valid), torch.from_numpy(x2_valid), torch.from_numpy(y_valid)]
    test_set = [x1_test, x2_test]

    return train_set, valid_set, test_set


def calc_bins(preds, labels, num_bins=10):
    # Assign each prediction to a bin
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for i in range(num_bins):
        bin_sizes[i] = len(preds[binned == i])
        if bin_sizes[i] > 0:
            bin_accs[i] = (labels[binned == i]).sum() / bin_sizes[i]
            bin_confs[i] = (preds[binned == i]).sum() / bin_sizes[i]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(preds, labels):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(preds, labels, figure_name):
    ECE, MCE = get_metrics(preds, labels)
    bins, _, bin_accs, _, _ = calc_bins(preds, labels)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    # plt.show()

    plt.savefig(figure_name, bbox_inches='tight')


def evaluate(model, dataloader, calibration_method=None, **kwargs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    preds = []
    labels = []
    correct = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1 = x1.to(device, non_blocking=True).float()
            x2 = x2.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            pred = model(x1, x2)

            if calibration_method:
                pred = calibration_method(pred, kwargs)

            # Get softmax values for net input and resulting class predictions
            sm = nn.Softmax(dim=1)
            pred = sm(pred)

            _, predicted_cl = torch.max(pred.data, 1)
            pred = pred.cpu().detach().numpy()
            preds.extend(pred[:, 1])

            labels.extend(y.detach().cpu().numpy())

            # Count correctly classified samples for accuracy
            correct += sum(predicted_cl == y).item()

    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    correct_perc = correct / len(dataloader.dataset)
    print('Accuracy of the network on the validation set: %d %%' % (100 * correct_perc))
    print(correct_perc)

    return preds, labels


def T_scaling(logits, kargs):
    temperature = kargs.get('temperature', None)
    return torch.div(logits, temperature)


def temperature_scaling(file_model, file_keys, file_data, path_out, batch_size=128):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        model = torch.load(file_model)
    else:
        model = torch.load(file_model, map_location="cpu")

    train_set, valid_set, test_set = get_datasets(file_keys, file_data)
    x1_train, x2_train, y_train = train_set
    x1_valid, x2_valid, y_valid = valid_set
    x1_test, x2_test = test_set

    dataset_train = torch.utils.data.TensorDataset(x1_train, x2_train, y_train)
    dataset_valid = torch.utils.data.TensorDataset(gaussian_noise(x1_valid), x2_valid, y_valid)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size)

    temperature = nn.Parameter(torch.ones(1).cuda())
    kargs = {'temperature': temperature}
    criterion = nn.CrossEntropyLoss()

    # Removing strong_wolfe line search results in jump after 50 epochs
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

    logits_list = []
    labels_list = []
    temps = []
    losses = []

    for x1, x2, y in dataloader_train:
        x1 = gaussian_noise(x1)
        x1 = x1.to(device, non_blocking=True).float()
        x2 = x2.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        model.eval()
        with torch.no_grad():
            logits_list.append(model(x1, x2))
            labels_list.append(y)

        # Create tensors
    logits_list = torch.cat(logits_list).to(device)
    labels_list = torch.cat(labels_list).to(device)

    def _eval():
        loss = criterion(T_scaling(logits_list, kargs), labels_list)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss.detach().cpu().numpy())
        return loss

    optimizer.step(_eval)

    print('Final T_scaling factor: {:.2f}'.format(temperature.item()))

    plt.subplot(121)
    plt.plot(list(range(len(temps))), temps)

    plt.subplot(122)
    plt.plot(list(range(len(losses))), losses)
    plt.show()

    preds_original, labels_original = evaluate(model, dataloader_valid)
    preds_calibrated, labels_calibrated = evaluate(model, dataloader_valid, T_scaling, temperature=temperature)

    draw_reliability_graph(preds_original, labels_original, os.path.join(path_out, "original.png"))
    draw_reliability_graph(preds_calibrated, labels_calibrated, os.path.join(path_out, "calibrated.png"))

    inference(x1_test, x2_test, model, temperature, path_out, nrow=5, ncol=4, npts=1000, batch_size=100)


def gaussian_noise(x, sig_dt=.4, sig_pot=.003):
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
    mask_zero = torch.where(x[:, :, 0] == 0, 0, 1)
    noise_dt = sig_dt * torch.randn((x.size(0), x.size(1)))
    # noise_pot = sig_pot * torch.randn((x.size(0), x.size(1)))
    noisy_data = x.clone()
    noisy_data[:, :, 0] += noise_dt * mask_pad * mask_zero
    # noisy_data[:, :, 1] += noise_pot * mask

    return noisy_data


def inference(samples, H0, model, temperature, path_out, nrow=5, ncol=4, npts=1000, batch_size=100):
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lower_bound = np.floor(np.min(H0))
    higher_bound = np.ceil(np.max(H0))

    # remove out of bounds data
    idx_up = np.where(H0 > 75.)[0]
    idx_down = np.where(H0 < 65.)[0]
    idx_out = np.concatenate((idx_up, idx_down))
    H0 = np.delete(H0, idx_out, axis=0)
    samples = np.delete(samples, idx_out, axis=0)
    nsamp = samples.shape[0]
    z = np.array([.5, 1.5])
    z = np.tile(z, (nsamp, 1))

    # observations
    x = gaussian_noise(torch.from_numpy(samples))
    data = torch.repeat_interleave(x, npts, dim=0)

    # Global NRE posterior
    gb_prior = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    gb_pr_tile = np.tile(gb_prior, (nsamp, 1))

    dataset_test = torch.utils.data.TensorDataset(data, torch.from_numpy(gb_pr_tile), torch.zeros((nsamp * npts)))
    dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    gb_probs, _ = evaluate(model.to(device), dataloader)  # , T_scaling, temperature=temperature)

    gb_ratios = gb_probs / (1 - gb_probs)
    gb_ratios = gb_ratios.reshape(nsamp, npts)
    gb_ratios = normalization(gb_pr_tile.reshape(nsamp, npts), gb_ratios)

    # local NRE posterior
    gb_prior = gb_prior.flatten()
    lc_prior = torch.zeros((nsamp, npts))
    true = np.zeros((nsamp,))
    for i in range(nsamp):
        true[i] = float(H0[i])
        max_prob = gb_prior[np.argmax(gb_ratios[i])]
        lc_prior[i] = torch.linspace(max_prob - 1., max_prob + 1., npts)

    dataset_loc = torch.utils.data.TensorDataset(data, lc_prior.reshape(nsamp * npts, 1), torch.zeros((nsamp * npts)))
    dataloader = torch.utils.data.DataLoader(dataset_loc, batch_size=batch_size)
    lc_probs, _ = evaluate(model.to(device), dataloader)  # , T_scaling, temperature=temperature)
    lc_ratios = lc_probs / (1 - lc_probs)

    # predictions
    arg_pred = np.argmax(lc_ratios.reshape(nsamp, npts), axis=1)
    pred = lc_prior[np.arange(nsamp), arg_pred].detach().cpu().numpy()

    # analytical posterior
    analytic = analytical_likelihood(x[:, :, 0].numpy(), x[:, :, 1].numpy(), gb_prior, z[:, 1], z[:, 0])
    analytic = normalization(gb_prior, analytic)

    it = 0
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=False, figsize=(3 * ncol, 3 * nrow))
    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].plot(gb_prior, analytic[it], '--g', label='{:.2f}'.format(lc_prior[it][np.argmax(analytic[it])]))
            axes[i, j].plot(gb_prior, gb_ratios[it], '-b', label='{:.2f}'.format(pred[it]))
            min_post = np.minimum(np.min(gb_ratios[it]), np.min(analytic[it]))
            max_post = np.maximum(np.max(gb_ratios[it]), np.max(analytic[it]))
            axes[i, j].vlines(true[it], min_post, max_post, colors='r', linestyles='dotted',
                              label='{:.2f}'.format(true[it]))
            axes[i, j].legend(frameon=False, borderpad=.2, handlelength=.6, fontsize=9, handletextpad=.4)
            if np.count_nonzero(samples[it, 0] + 1) == 3:
                axes[i, j].set_title("Quad")
            if np.count_nonzero(samples[it, 0] + 1) == 1:
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

    NRE_lc = post_file.create_group("NRE_local")
    H0_lc = NRE_lc.create_dataset("H0", (nsamp, npts), dtype='f')
    post_lc = NRE_lc.create_dataset("posterior", (nsamp, npts), dtype='f')

    NRE_gb = post_file.create_group("NRE_global")
    H0_gb = NRE_gb.create_dataset("H0", (npts,), dtype='f')
    post_gb = NRE_gb.create_dataset("posterior", (nsamp, npts), dtype='f')
    post_anl = NRE_gb.create_dataset("analytical", (nsamp, npts), dtype='f')

    truth_set = post_file.create_dataset("truth", (nsamp,), dtype='f')

    H0_lc[:, :] = lc_prior
    post_lc[:, :] = lc_ratios.reshape(nsamp, npts)
    H0_gb[:] = gb_prior
    post_gb[:, :] = gb_ratios
    post_anl[:, :] = analytic
    truth_set[:] = true

    post_file.close()

    # integration from pred to true
    start = np.minimum(true, pred)
    end = np.maximum(true, pred)
    interval = np.zeros((nsamp,))
    for i in range(nsamp):
        idx_up = np.where(gb_prior > end[i])[0]
        idx_down = np.where(gb_prior < start[i])[0]
        idx_out = np.concatenate((idx_up, idx_down))
        interval_x = np.delete(gb_prior, idx_out)
        interval_y = np.delete(gb_ratios[i], idx_out)
        interval[i] = np.trapz(interval_y, interval_x)

    # Coverage diagnostic
    bins = np.linspace(0., 1., 100)
    counts = np.zeros((100,))
    for i in range(len(bins)):
        counts[i] = np.sum(np.where(2 * interval <= bins[i], 1, 0))
    counts /= nsamp

    plt.style.use(['dark_background'])
    plt.figure()
    plt.plot(bins, bins, '--', color='white')
    plt.plot(bins, counts, '-', color='lime')
    plt.xlabel("Probability interval")
    plt.ylabel("Fraction of truths inside")
    plt.text(0., .9, "Underconfident", fontsize='large')
    plt.text(.65, .05, "Overconfident", fontsize='large')
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['savefig.facecolor'] = 'white'
    plt.savefig(path_out + '/coverage.png', bbox_inches='tight')


#############################################################################

# --- Execution ------------------------------------------------------------
if __name__ == "__main__":
    # --- Training ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train a classifier to be an estimator of the likelihood ratio of H_0")
    parser.add_argument("--path_data", type=str, default="", help="path to data")
    parser.add_argument("--model_file", type=str, default="", help="path and name of the trained model")
    parser.add_argument("--path_out", type=str, default="", help="path to save the outputs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")

    args = parser.parse_args()

    temperature_scaling(args.model_file,
                        os.path.join(args.path_data, "keys.hdf5"),
                        os.path.join(args.path_data, "dataset.hdf5"),
                        args.path_out,
                        batch_size=args.batch_size)