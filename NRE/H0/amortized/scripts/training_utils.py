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