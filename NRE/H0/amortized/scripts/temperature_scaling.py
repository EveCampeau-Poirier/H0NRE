import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tqdm
from functions import gaussian_noise

# Allow reproducability
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
    train_keys = np.random.choice(keys, size=int(.9 * nsamp), replace=False)
    test_keys = np.setdiff1d(keys, train_keys)

    # Reading data
    dataset = h5py.File(file_data, 'r')
    dt = dataset["time_delays"][calib_keys]
    pot = dataset["Fermat_potential"][calib_keys]
    H0 = dataset["Hubble_cst"][calib_keys]
    dataset.close()
    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)

    x1_train, x2_train = samples[train_keys], H0[train_keys]
    x1_test, x2_test = samples[test_keys], H0[test_keys]

    x2_train, y_train = get_labels(x2_train)
    x2_test, y_test = get_labels(x2_test)

    train_set = [torch.from_numpy(x1_train), torch.from_numpy(x2_train), torch.from_numpy(y_train)]
    test_set = [torch.from_numpy(x1_test), torch.from_numpy(x2_test), torch.from_numpy(y_test)]

    return train_set, test_set


def calc_bins(preds, labels, num_bins = 10):
  # Assign each prediction to a bin
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

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
    preds = []
    labels = []
    correct = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1 = gaussian_noise(x1)
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
            preds.extend(pred)

            labels.extend(y.detach().cpu().numpy())

            # Count correctly classified samples for accuracy
            correct += sum(predicted_cl == y).item()

    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    correct_perc = correct / len(dataloader.dataset)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_perc))
    print(correct_perc)

    return preds, labels


def T_scaling(logits, args):
    temperature = args.get('temperature', None)
    return torch.div(logits, temperature)


def calibration(file_model, file_keys, file_data, batch_size=128):
    if torch.cuda.is_available():
        model = torch.load(file_model)
    else:
        model = torch.load(file_model, map_location="cpu")

    train_set, test_set = get_datasets(file_keys, file_data)
    x1_train, x2_train, y_train = train_set
    x1_test, x2_test, y_test = test_set
    dataset_train = torch.utils.data.TensorDataset(x1_train, x2_train, y_train)
    dataset_test = torch.utils.data.TensorDataset(x1_test, x2_test, y_test)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    temperature = nn.Parameter(torch.ones(1).cuda())
    args = {'temperature': temperature}
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
        loss = criterion(T_scaling(logits_list, args), labels_list)
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

    preds_original, _ = evaluate(model, dataloader_test)
    preds_calibrated, _ = evaluate(model, dataloader_test, T_scaling, temperature=temperature)

    draw_reliability_graph(preds_original, labels, "original.png")
    draw_reliability_graph(preds_calibrated, labels, "calibrated.png")


#############################################################################

# --- Execution ------------------------------------------------------------
if __name__ == "__main__":

    # --- Training ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train a classifier to be an estimator of the likelihood ratio of H_0")
    parser.add_argument("--path_data", type=str, default="", help="path to data")
    parser.add_argument("--model_file", type=str, default="", help="path and name of the trained model)
    parser.add_argument("--path_out", type=str, default="", help="path to save the outputs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--nepochs", type=int, default=100, help="number of training epochs")

    args = parser.parse_args()

    calibration(args.model_file,
                os.path.join(args.path_data, "keys.hdf5"),
                os.path.join(args.path_data, "dataset.hdf5"),
                batch_size=args.batch_size)