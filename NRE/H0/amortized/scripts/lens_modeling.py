# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
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

def linear(x, a, b):
    return a * x + b


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
    images = dataset["images"][:]
    param = dataset["parameters"][:, :7]
    dataset.close()

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

    return train_set, valid_set, mean, std


def train_fn(model, file, path_in, path_out, optimizer, loss_fn, acc_fn, threshold, sched=None,
             grad_clip=None, anomaly_detection=False, batch_size=256, epochs=100, std_noise=.5):
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
    train_set, valid_set, mean, std = split_data(file, path_in)
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

    return mean, std


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


def evaluation(file_keys, file_data, file_model, path_out, mean, std, nrow=2, ncol=4):
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
    truths = dataset["parameters"][test_keys, :7]
    dataset.close()
    truths[:, 2], truths[:, 3] = ellip_coordinates(truths[:, 2], truths[:, 3])
    truths[:, 5], truths[:, 6] = shear_coordinates(truths[:, 5], truths[:, 6])

    x = images.to(device, non_blocking=True).float()
    y = truths.to(device, non_blocking=True).float()
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

    ticks = np.array([[-.4, .4, .2], [-.4, .4, .2], [-1., 1., .5], [-1., 1., .5],
                      [.0, 2.5, .5], [-.07, .07, .05], [-.07, .07, .05]])

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
    plt.savefig(path_out + '/acc.png', bbox_inches='tight')