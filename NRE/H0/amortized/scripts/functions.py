# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

from simulator import training_set
ts = training_set()

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


def noise(x, expo_time=1000, sig_bg=.001):
    """
    Adds noise to images
    Inputs
        x : (tensor)[batch_size x nchan x npix x npix] image
        expo_time : (float) exposure time
        sig_bg : (float) standard deviation of background noise
    Outputs
        noisy_im : (tensor)[batch_size x nchan x npix x npix] noisy image
    """
    poisson = sig_bg*torch.randn(x.size()) # poisson noise
    bckgnd = torch.sqrt(abs(x)/expo_time)*torch.randn(x.size()) # background noise
    noisy_im = bckgnd+poisson+x
    
    return noisy_im


def gaussian_noise(x, sig_dt=.3, sig_pot=.003):
    """
    Adds noise to time delays
    Inputs
        x : (tensor)[batch_size x 4 x 2] Time delays and Fermat potentials
        sig_dt : (float) noise standard deviation on time delays
    Outputs
        noisy_data : (tensor)[batch_size x 4 x 2] noisy time delays + true Fermat potential
    """
    mask_zero = torch.where(x[:, :, 0] == 0, 0, 1)
    mask_pad = torch.where(x[:, :, 0] == -1, 0, 1)
    noise_dt = sig_dt * torch.randn((x.size(0), x.size(1)))
    noise_pot = sig_pot * torch.randn((x.size(0), x.size(1)))
    noisy_data = x.clone()
    noisy_data[:, :, 0] += noise_dt * mask_zero * mask_pad
    noisy_data[:, :, 1] += noise_pot * mask_zero * mask_pad

    return noisy_data


def analytical_likelihood(dt, pot, H0, zs, zd, sigma=.4):
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
    lkh = np.exp(-np.sum((dt[:, None] - mu) ** 2, axis=2) / 2 / sigma ** 2) / (2 * np.pi * sigma ** 2) ** size[:, None]

    return lkh

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
    logp = np.delete(logp,ind_max)
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
        
    prob = torch.sigmoid(output[:,1])
    s = prob.detach().cpu().numpy()
    lr = s/(1-s)
    
    return lr


def find_index(tensor, value):
    """
    Find index of the closest tensor element to value
    Inputs
        tensor : (tensor) torch tensor
        value : (float) value
    Outputs
        ind : (int) index
    """
    diff = abs(tensor - value)
    ind = torch.where(diff == torch.min(diff))[0][0]
    
    return ind


def index_probe(theta, labels, probe, dependence=True):
    """
    Find index of the tensor element closest to the probe value in a given class
    Inputs
        theta : (tensor) parameter to probe
        labels : (tensor) Labels associated with theta
        probe : (float) Monitored parameter value
        dependence : (bool) Class to probe
    Outputs
        idx : (int) index
    """
    if dependence:
        ids = torch.nonzero(labels)
    else:
        ids = torch.nonzero(labels-1)
    class_val = theta[ids]
    id_class = find_index(class_val, probe)
    idx = ids[id_class]

    return idx


def normalization(x, y):
    """
        Normalizes the analytical posterior
        Inputs
            x : (array) H0 values
            y : (array) posterior probabilities
        Outputs
            y : (array) Norm
        """
    norm = np.trapz(y, x, axis=1)
    y /= norm[:, None]

    return y


def make_classes(x1, x2, lower_bound, higher_bound):
    """
    Creates labels, shuffles data
    Inputs
        x1 : (array) [nexamp x 4 x 2] data tensor
        x2 : (array) [nexamp x 1] H_0 tensor
    Outputs
        x1 : (tensor) [nexamp x 4 x 2] data tensor
        x2 : (tensor) [nexamp x 1] H_0 tensor
        y : (tensor) [nexamp x 1] label tensor
    """
    nsamp = x1.shape[0]
    nclass = int(nsamp / 2)
    if nsamp % 2 == 0:
        x2[nclass:] = np.random.uniform(lower_bound, higher_bound, size=(nclass, 1))
        y = np.concatenate((np.ones(nclass), np.zeros(nclass)))
    else:
        x2[nclass + 1:] = np.random.uniform(lower_bound, higher_bound, size=(nclass, 1))
        y = np.concatenate((np.ones(nclass + 1), np.zeros(nclass)))

    shuffle = np.random.choice(nsamp, nsamp, replace=False)
    x1 = torch.from_numpy(x1[shuffle])
    x2 = torch.from_numpy(x2[shuffle])
    y = torch.from_numpy(y[shuffle])

    return x1, x2, y


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
    lower_bound = np.floor(np.min(H0))
    higher_bound = np.ceil(np.max(H0))

    # Splitting sets
    nsamp = samples.shape[0]
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

    # Making labels + torch format
    x1_train, x2_train, y_train = make_classes(x1_train, x2_train, lower_bound, higher_bound)
    x1_valid, x2_valid, y_valid = make_classes(x1_valid, x2_valid, lower_bound, higher_bound)

    # Outputs
    train_set = [x1_train, x2_train, y_train]
    valid_set = [x1_valid, x2_valid, y_valid]

    return train_set, valid_set


def train_fn(model, file, path_in, path_out, optimizer, loss_fn, acc_fn, threshold, sched=None,
             grad_clip=None, anomaly_detection=False, batch_size=128, epochs=100, probe=70):
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
        nprobes : (int) number of likelihood ratio probes
    Outputs
        None
    """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # model and loss function on GPU
    model = model.to(device, non_blocking=True)
    loss_fn = loss_fn.to(device, non_blocking=True)
    
    # Datasets
    train_set, valid_set = split_data(file, path_in)
    x1_train, x2_train, y_train = train_set
    x1_val, x2_val, y_val = valid_set
    
    dataset_train = torch.utils.data.TensorDataset(x1_train, x2_train, y_train)
    dataset_valid = torch.utils.data.TensorDataset(x1_val, x2_val, y_val)

    # probe indices
    idx_dep_train = index_probe(x2_train, y_train, probe)
    idx_ind_train = index_probe(x2_train, y_train, probe, dependence=False)
    idx_train = [int(idx_dep_train), int(idx_ind_train)]
    
    idx_dep_val = index_probe(x2_val, y_val, probe)
    idx_ind_val = index_probe(x2_val, y_val, probe, dependence=False)
    idx_val = [int(idx_dep_val), int(idx_ind_val)]
    
    # File to save logs
    if os.path.isfile(path_out+'/logs.hdf5'):
        os.remove(path_out+'/logs.hdf5')
    save_file = h5py.File(path_out+'/logs.hdf5','a')
    
    trng = save_file.create_group("training")
    train_loss = trng.create_dataset("loss", (epochs, 1), dtype='f')
    train_acc = trng.create_dataset("accuracy", (epochs, 1), dtype='f')
    train_lr = trng.create_dataset("likelihood_ratio", (epochs, 2), dtype='f')
    
    vldt = save_file.create_group("validation")
    valid_loss = vldt.create_dataset("loss", (epochs, 1), dtype='f')
    valid_acc = vldt.create_dataset("accuracy", (epochs, 1), dtype='f')
    valid_lr = vldt.create_dataset("likelihood_ratio", (epochs, 2), dtype='f')
    
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
                x1_probe, x2_probe = x1_train[idx_train], x2_train[idx_train]
            else:
                model.train(False)  # evaluation
                dataloader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size)
                x1_probe, x2_probe = x1_val[idx_val], x2_val[idx_val]
                
            # Initialization of cumulative loss on batches
            running_loss = 0.0
            # Initialization of cumulative accuracy on batches
            running_acc = 0.0
            
            step = 0   # step initialization (batch number)
            
            # loop on batches
            for x1, x2, y in dataloader:
                x1 = gaussian_noise(x1)
                x1 = x1.to(device, non_blocking=True).float()
                x2 = x2.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).long()
                
                # training phase
                if phase == 'train':
                    # Forward pass
                    for param in model.parameters():
                        param.grad = None
                    
                    if anomaly_detection:
                        with autograd.detect_anomaly():
                            y_hat = model(x1, x2)
                            loss = loss_fn(y_hat, y)
                    else:
                        y_hat = model(x1, x2)
                        loss = loss_fn(y_hat, y)
                    
                    # Backward Pass
                    loss.backward()
                    
                    if grad_clip is not None:
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        
                    optimizer.step()
                    
                # validation phase
                else:
                    with torch.no_grad():
                        y_hat = model(x1, x2)
                        loss = loss_fn(y_hat, y)
                
                # accuracy evaluation
                acc = acc_fn(y_hat, y)
                
                # update cumulative values
                running_acc += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size
                
                # print current information
                if step % int(len(dataloader.dataset)/batch_size/20) == 0: #
                    print(f'Current {phase} step {step} ==>  Loss: {float(loss):.4e} // Acc: {float(acc):.4e} // AllocMem (Gb): {torch.cuda.memory_reserved(0)*1e-9}') 
                
                step += 1
                
            # mean of loss and accuracy on epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            epoch_lr = r_estimator(model, x1_probe, x2_probe)

            # print means
            print(f'{phase} Loss: {float(epoch_loss):.4e} // Acc: {float(epoch_acc):.4e}')
            
            # append means to lists
            if phase == 'train':
                train_loss[epoch, :] = epoch_loss.detach().cpu().numpy()
                train_acc[epoch, :] = epoch_acc.detach().cpu().numpy()
                train_lr[epoch, :] = epoch_lr
                if sched is not None and epoch <= threshold:
                    sched.step()
            else:
                valid_loss[epoch, :] = epoch_loss.detach().cpu().numpy()
                valid_acc[epoch, :] = epoch_acc.detach().cpu().numpy()
                valid_lr[epoch, :] = epoch_lr

        # Keeping track of the model
        if epoch % int(epochs/10) == 0:
            torch.save(model, path_out+f'/models/model{epoch:02d}.pt')
    
    # Closing file
    save_file.close()
    
    # print training time
    time_elapsed = time.time() - start
    print(f'Training completed in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')



# --- plots ----------------------------------------------
def plot_results(file, path_out):
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
    train_lr = data["training/likelihood_ratio"][:]
    valid_lr = data["validation/likelihood_ratio"][:]
    data.close()
    
    # loss
    plt.figure()
    plt.plot(train_loss, 'b', label='Training')
    plt.plot(valid_loss, 'r', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross entropy)")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.legend()
    plt.savefig(path_out+'/loss.png', bbox_inches='tight')

    # accuracy
    plt.figure()
    plt.plot(100*train_acc, 'b', label='Training')
    plt.plot(100*valid_acc, 'r', label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.legend()
    plt.savefig(path_out+'/acc.png', bbox_inches='tight')
    
    # Likelihood ratio
    color = ["green", "darkorange", "lime", "darkviolet", "cyan",
             "deeppink", "darkblue", "gold", "maroon"] # "red","blue",
    labels = ["dependent", "independent"]
    plt.figure()
    for i in range(train_lr.shape[1]):
        plt.plot(train_lr[:, i], color=color[i], label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Likelihood ratio estimation")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.title(r'H$_0$ = 70 km s$^{-1}$ Mpc$^{-1}$')
    plt.legend()
    plt.savefig(path_out+'/train_lr.png', bbox_inches='tight')

    plt.figure()
    for i in range(valid_lr.shape[1]):
        plt.plot(valid_lr[:, i], color=color[i], label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel("Likelihood ratio estimation")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.title(r'H$_0$ = 70 km s$^{-1}$ Mpc$^{-1}$')
    plt.legend()
    plt.savefig(path_out+'/valid_lr.png', bbox_inches='tight')

    
def inference(file_keys, file_data, file_model, path_out, nrow=5, ncol=4, npts=1000):
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
    nplot = int(nrow * ncol)

    # import keys
    keys = h5py.File(file_keys, 'r')
    test_keys = keys["test"][:]
    keys.close()

    # import data
    dataset = h5py.File(file_data, 'r')
    H0 = dataset["Hubble_cst"][test_keys]
    lower_bound = np.floor(np.min(H0))
    higher_bound = np.ceil(np.max(H0))
    # remove out of bounds data
    idx_up = np.where(H0 > 75.)[0]
    idx_down = np.where(H0 < 65.)[0]
    idx_out = np.concatenate((idx_up, idx_down))
    H0 = np.delete(H0, idx_out, axis=0)
    test_keys = np.delete(test_keys, idx_out, axis=0)
    dt = dataset["time_delays"][test_keys]
    pot = dataset["Fermat_potential"][test_keys]
    z = dataset["redshifts"][test_keys]
    dataset.close()
    samples = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)
    samples = samples[:1000]
    nsamp = samples.shape[0]

    # observations
    x = gaussian_noise(torch.from_numpy(samples))
    data = torch.repeat_interleave(x, npts, dim=0)

    # Global NRE posterior
    gb_prior = np.linspace(lower_bound + 1, higher_bound - 1, npts).reshape(npts, 1)
    gb_pr_tile = np.tile(gb_prior, (nsamp, 1))
    gb_ratios = r_estimator(model, data, torch.from_numpy(gb_pr_tile))

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
    lc_ratios = r_estimator(model, data, lc_prior.reshape(npts * nsamp, 1))

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
            axes[i, j].plot(gb_prior, analytic[it], '--g', label="Analytic")
            axes[i, j].plot(gb_prior, gb_ratios[it], '-b', label='{:.2f}'.format(pred[it]))
            min_post = np.minimum(np.min(gb_ratios[it]), np.min(analytic[it]))
            max_post = np.maximum(np.max(gb_ratios[it]), np.max(analytic[it]))
            axes[i, j].vlines(true[it], min_post, max_post, colors='r', linestyles='dotted',
                              label='{:.2f}'.format(true[it]))
            axes[i, j].legend(frameon=False, borderpad=.2, handlelength=.6, fontsize=9, handletextpad=.4)
            if np.count_nonzero(samples[it, 0] + 1) == 4:
                axes[i, j].set_title("Quad")
            if np.count_nonzero(samples[it, 0] + 1) == 2:
                axes[i, j].set_title("Double")
            # axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

            if i == int(nrow - 1):
                axes[i, j].set_xlabel(r"H$_0$ (km Mpc$^{-1}$ s$^{-1}$)")
            if j == 0:
                axes[i, j].set_ylabel("Likelihood ratio")

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
    H0_lc = NRE_lc.create_dataset("H0", (nplot, npts), dtype='f')
    post_lc = NRE_lc.create_dataset("posterior", (nplot, npts), dtype='f')

    NRE_gb = post_file.create_group("NRE_global")
    H0_gb = NRE_gb.create_dataset("H0", (nsamp,), dtype='f')
    post_gb = NRE_gb.create_dataset("posterior", (nplot, npts), dtype='f')
    post_anl = NRE_gb.create_dataset("analytical", (nplot, npts), dtype='f')

    truth_set = post_file.create_dataset("truth", (nplot,), dtype='f')

    H0_lc[:, :] = lc_prior[:nplot]
    post_lc[:, :] = lc_ratios.reshape(nsamp, npts)[:nplot]
    H0_gb[:] = gb_prior
    post_gb[:, :] = gb_ratios[:nplot]
    post_anl[:, :] = analytic[:nplot]
    truth_set[:] = true[:nplot]

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

    plt.figure()
    plt.plot(bins, bins, '--k')
    plt.plot(bins, counts, '-g')
    plt.xlabel("Probability interval")
    plt.ylabel("Fraction of truths inside")
    plt.text(0., .9, "Underconfident")
    plt.text(.65, .05, "Overconfident")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.savefig(path_out + '/coverage.png', bbox_inches='tight')


def joint_inference(file_data, file_model, path_out, npts=1000):
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
    true = float(np.unique(H0))

    # observations
    x = gaussian_noise(torch.from_numpy(samples))
    data = torch.repeat_interleave(x, npts, dim=0)

    # Global NRE posterior
    prior = np.linspace(65, 75, npts).reshape(npts, 1)
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
