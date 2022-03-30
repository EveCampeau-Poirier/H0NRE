# --- Importations ---------------------------------------------------------

# Python libraries
import time
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split

# PyTorch libraries
import torch
from torch.nn.utils import clip_grad_norm_
from torch import autograd


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
    bckgrd = torch.sqrt(abs(x)/expo_time)*torch.randn(x.size()) # bakcground noise
    noisy_im = bckgrd+poisson+x
    
    return noisy_im


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


def make_sets(x1, x2, save=False, file_name=""):
    """
    Creates labels, shuffles data
    Inputs 
        x1 : (array) [nexamp x 4] Time delay tensor
        x2 : (array) [nexamp x 1] H_0 tensor
    Outputs
        x1 : (tensor) [nexamp x 4] Time delay tensor
        x2 : (tensor) [nexamp x 1] H_0 tensor
        y : (tensor) [nexamp x 1] label tensor
    """
    if save :
        if os.path.isfile(file_name):
            os.remove(file_name)
        nsamp = x1.shape[0]
        file = h5py.File(file_name, 'a')
        dt_set = file.create_dataset("time_delays", (nsamp, 4), dtype='f')
        pot_set = file.create_dataset("Fermat_potential", (nsamp, 4), dtype='f')
        H0_set = file.create_dataset("Hubble_cst", (nsamp, 1), dtype='f')
        dt_set[:,:] = x1[:,:,0]
        pot_set[:,:] = x1[:,:,1]
        H0_set[:,:] = x2
        file.close()
    
    nsamp = x1.shape[0]
    nclass = int(nsamp/2)
    if nsamp %2 == 0:
        x2[nclass:] = np.random.uniform(64., 76., size=(nclass,1))
        y = np.concatenate((np.ones(nclass), np.zeros(nclass)))
    else : 
        x2[nclass+1:] = np.random.uniform(64., 76., size=(nclass,1))
        y = np.concatenate((np.ones(nclass+1), np.zeros(nclass)))
                            
    shuffle = np.random.choice(nsamp, nsamp, replace=False)
    x1 = torch.from_numpy(x1[shuffle])
    x2 = torch.from_numpy(x2[shuffle])
    y = torch.from_numpy(y[shuffle])
                       
    return x1, x2, y


def split_data(file, path_in):
    """
    Reads data, preprocesses data and splits data into training, validation and test sets
    Inputs 
        file : (str) name of the file containing data
    Outputs
        All outputs are a list of the images, the time delays, the H0 and the labels tensors
        train_set : (list) training set. Corresponds to 60% of all data.
        valid_set : (list) validation set. Corresponds to 20% of all data.
        test_set : (list) test set. Corresponds to 20% of all data.
    """
    dataset = h5py.File(os.path.join(path_in, file), 'r')
    
    # Time delays
    dt = dataset["time_delays"][:]

    # Fermat potential
    pot = dataset["Fermat_potential"][:]
    
    # Hubble constant
    H0 = dataset["Hubble_cst"][:]
    
    # close files
    dataset.close()

    images = np.concatenate((dt[:, :, None], pot[:, :, None]), axis=2)
    
    # Splitting training and test sets
    x1_train, x1_test, x2_train, x2_test = train_test_split(images, H0, test_size=0.20, random_state=11)

    # Test set in torch format
    x1_test, x2_test, y_test = make_sets(x1_test, x2_test, save=True, file_name=os.path.join(path_in, "test_set.hdf5"))
    
    # Splitting training and validation sets
    x1_train, x1_val, x2_train, x2_val = train_test_split(x1_train, x2_train, test_size=0.25, random_state=11)

    # Training set in torch format
    x1_train, x2_train, y_train = make_sets(x1_train, x2_train)

    # Validation set in torch format
    x1_val, x2_val, y_val = make_sets(x1_val, x2_val)
    
    # Outputs
    train_set = [x1_train, x2_train, y_train]
    valid_set = [x1_val, x2_val, y_val]
    test_set = [x1_test, x2_test, y_test]
    
    return train_set, valid_set, test_set


def train_fn(model, file, path_in, path_out, optimizer, loss_fn, acc_fn, sched=None, 
             grad_clip=None, anomaly_detection=False, batch_size=128, epochs=60, probe=70):
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
        sched : If not None, learning rate scheduler to update
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
    train_set, valid_set, test_set = split_data(file, path_in)
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
                x1 = x1.to(device, non_blocking=True).float()
                x2 = x2.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).float()
                
                # training phase
                if phase == 'train':
                    # Forward pass
                    for param in model.parameters():
                        param.grad = None
                    
                    if anomaly_detection:
                        with autograd.detect_anomaly():
                            y_hat = model(x1, x2)
                            loss = loss_fn(y_hat, y.long())
                    else:
                        y_hat = model(x1, x2)
                        loss = loss_fn(y_hat, y.long())
                    
                    # Backward Pass
                    loss.backward()
                    
                    if grad_clip is not None:
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        
                    optimizer.step()
                    
                # validation phase
                else:
                    with torch.no_grad():
                        y_hat = model(x1, x2)
                        loss = loss_fn(y_hat, y.long())
                
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
                if sched is not None:
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

    
def inference(file_test, file_model, path_out, nrow=5, ncol=4, npts=1000):
    
    model = torch.load(file_model)
    
    test_set = h5py.File(file_test, 'r')
    time_delays = test_set["time_delays"][:]
    fermat_pot = test_set["Fermat_potential"][:]
    H0 = test_set["Hubble_cst"][:]
    test_set.close()

    images = np.concatenate((time_delays[:, :, None], fermat_pot[:, :, None]), axis=2)

    idx_up = np.where(H0 >= 75)[0]
    idx_down = np.where(H0 <= 65)[0]
    idx_out = np.concatenate((idx_up, idx_down))
    H0 = np.delete(H0, idx_out, axis=0)
    images = np.delete(images, idx_out, axis=0)
    
    prior = torch.linspace(65, 75, npts)
    prior = prior.reshape(npts, 1)
    
    it = 0
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, sharex=True, sharey=False, figsize=(3*ncol, 3*nrow))
    for i in range(nrow):
        for j in range(ncol):
            im = np.tile(images[it], (npts, 1, 1))
            im = torch.from_numpy(im)
            ratios = r_estimator(model, im, prior)
            
            true = float(H0[it])
            pred = float(prior[np.argmax(ratios)])
            
            axes[i, j].plot(prior, ratios, 'b', label='{:.2f}'.format(pred))
            axes[i, j].vlines(true, np.min(ratios), np.max(ratios), colors='r', label='{:.2f}'.format(true))
            axes[i, j].legend(frameon=False, borderpad=.2, handlelength=.6, fontsize=9, handletextpad=.4)
            axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
            
            if i == int(nrow-1):
                axes[i, j].set_xlabel(r"H$_0$ (km Mpc$^{-1}$ s$^{-1}$)")
            if j == 0:
                axes[i, j].set_ylabel("Likelihood ratio")
            
            it += 1
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.savefig(path_out+'/posteriors.png', bbox_inches='tight')