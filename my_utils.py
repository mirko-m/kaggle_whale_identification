import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
import os.path
import time
import PIL
import matplotlib.pyplot as plt
import copy

# import skimage
# import skimage.io


MEAN = [0.485, 0.456, 0.406] # expected by pretrained resnet18
STD = [0.229, 0.224, 0.225] # expected by pretrained resnet18

class WhaleDataset(torch.utils.data.Dataset):

    def __init__(self,data_frame,data_dir,transform=None):
        '''
        Parameters
        ----------
        data_frame: (pandas data_frame) contains the names of the images and
            the corresponding labels in the 0th and 1st column
        data_dir: (string) path to the data directory

        __getitem__ method returns
        --------------------------
        image: (PIL image) .
        label: (integer) label for the image

        Note
        ----
        1. To convert integer labels to strings use the categories attribute

        2. Use pytorch transform.ToTensor() to transform the images.

        '''
        # data_frame = pandas.read_csv(os.path.join(data_dir,csv_file))
        categories = list(set(c for c in data_frame.iloc[:,1]))
        categories.sort()
        self.data_frame = data_frame
        self.data_dir = data_dir
        self.categories = categories
        self.transform = transform

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, index):
        image_name = os.path.join(self.data_dir,'train',\
                                  self.data_frame.iloc[index,0])

        image = PIL.Image.open(image_name)
        # some of the images are gray-scale only. Convert them to RGB. This
        # basically just copies the image 3 times.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform != None:
            image = self.transform(image)

        label = self.categories.index(self.data_frame.iloc[index,1])
        return image, label

def predict(model,test_dataloader,device,categories):
    '''Still need to test this

    Parameters
    ----------
    model: trained model
    test_dataloader: pytorch dataloader for test-data
    device: torch.device
    categories: list with categories used for training, i.e.
        taining_dataset.categories

    Returns
    -------
    df: pandas DataFrame with the top 5 predictions in the required format.

    '''

    size = float(len(dataloader.sampler))
    Id_vals = []
    Image_vals = []
    cols = ['Image', 'Id']
    result = []

    # iterate over data
    for images, image_names in test_dataloader:
        images = images.to(device)

        #forward
        out = model(images)
        _, preds = torch.topk(out,k=5,dim=1)
        torch.cat(image_names,preds,dim=1)

        preds = preds.tolist()
        for j in xrange(batch_size):
            Image_vals.append(image_names[j])
            Id_vals.append(''.join([categories[i]+' ' for i in preds[j]]))
        df = pd.DataFrame({'Image':Image_vals,\
                           'Id':Id_vals})

    return df

def _val_loop(model,dataloader,criterion,device):
    '''
    Helper function implementing the validation loop

    Parameters
    ----------
    model: the pytorch model to be trained
    dataloader_dict: dictionary with keys 'train' and 'val' (optional)
        containing pytorch DataLoader instances.
    criterion: The pytorch criterion for the loss (loss-function)
    device: torch.device

    Returns
    -------
    loss: (float) the loss
    acc:  (float) the accuracy
    '''

    size = float(len(dataloader.sampler))
    running_loss = 0.0
    running_corrects = 0.0

    # iterate over data
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        #forward
        out = model(images)
        preds = torch.argmax(out,1)
        loss = criterion(out,labels)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / size
    acc = float(running_corrects) / size
    return loss, acc

def _train_loop(model,dataloader,criterion,optimizer,device):
    '''
    Helper function implementing the validation loop

    Parameters
    ----------
    model: the pytorch model to be trained
    dataloader_dict: dictionary with keys 'train' and 'val' (optional)
        containing pytorch DataLoader instances.
    optimizer: pytorch Optimizer instance.
    criterion: The pytorch criterion for the loss (loss-function)
    device: torch.device

    Returns
    -------
    loss: (float) the loss
    acc:  (float) the accuracy
    '''

    size = float(len(dataloader.sampler))
    running_loss = 0.0
    running_corrects = 0.0

    # iterate over data
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        #  track history only when in train
        with torch.set_grad_enabled(True):
            #forward
            out = model(images)
            preds = torch.argmax(out,1)
            loss = criterion(out,labels)

            # backward
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / size
    acc = float(running_corrects) / size
    return loss, acc

def find_learning_rate(model,dataloader,criterion,optimizer,device,\
                       lr_vals=None,verbose=True):
    '''Train the model while increasing the learning rate. Use this function to
    find a good value for the initial learning rate. This is a trick I learned
    from the fast.ai MOOC.

     Parameters
    ----------
    model: the pytorch model to be trained
    dataloader:  pytorch DataLoader instance.
    criterion: The pytorch criterion for the loss (loss-function)
    optimizer: pytorch Optimizer instance.
    device: torch.device
    lr_vals: list with values for the learning rate if None is passed a default
        list is used.
    verbose: (default + True) whether to print status updates

    Returns
    -------
    best_model: best model (according to validation accuracy or training
        accuracy if use_val is False)
    lr_vals: list with the learning rate values. If lr_vals=None is passed the
        default learning ratevalues are returned.
    loss_vals: list with loss values
    acc_vals: list with accuracy values

    Note
    ----
    Check whether the behavior corresponds to model being passed "as copy" or
    "by reference"

    '''

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # default values for lr_vals
    if lr_vals == None:
        lr_vals = list(0.0001*1.2**np.array(range(18)))

    loss_vals = []
    acc_vals = []

    t_start = time.time()

    for lr in lr_vals:

        # set the learning rate of the optimizer
        for param in optimizer.param_groups:
                    param['lr'] = lr

        model.train() # Set model to training mode
        loss, acc = _train_loop(model,dataloader,criterion,optimizer,device)
        loss_vals.append(loss)
        acc_vals.append(acc)
        t = time.time()

        if verbose:
            print 'Elapsed time: {:.4f} lr: {:.4e} Loss: {:.4f} Acc: {:.4f}'.\
                format(t - t_start, lr, loss, acc)

        if  acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

    t_final = time.time()
    if verbose:
        print 'Total time {:.4f}'.format(t_final - t_start)

    model.load_state_dict(best_model_wts)
    return model, lr_vals, loss_vals, acc_vals

def train_with_restart(model,dataloader_dict,criterion,optimizer,device,\
                       T_max=10,num_epochs=20,verbose=True,use_val=True,\
                       save_prog=True):
    '''Train the model with a cosine-shaped learning rate annealer and restarts
    after T_max epochs. This is a trick I picked up from the fast.ai MOOC.

    Parameters
    ----------
    model: the pytorch model to be trained
    dataloader_dict: dictionary with keys 'train' and 'val' (optional)
        containing pytorch DataLoader instances.
    criterion: The pytorch criterion for the loss (loss-function)
    optimizer: pytorch Optimizer instance.
    device: torch.device
    T_max: (default = 10) learning rate is reset to initial value after T_max
        epochs.
    num_epochs: (default = 20)
    verbose: (default + True) whether to print status updates
    use_val: (default = True) set to False if no validation Dataloader is
        passed.
    save_prog: (default=True) periodically save the model to the file
        train_with_restart_progress.pt

    Returns
    -------
    best_model: best model (according to validation accuracy or training
        accuracy if use_val is False)
    loss: dict with loss values
    acc: dict with accuracy values

    Note
    ----
    Check whether the behavior corresponds to model being passed "as copy" or
    "by reference"

    '''

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    prog_fname = 'train_with_restart_progress.pt'

    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}

    t_start = time.time()

    for epoch in range(num_epochs):

        if verbose:
            print 'Epoch {:d} / {:d}'.format(epoch,num_epochs)

        # restart after T_max epochs
        if epoch % scheduler.T_max == 0:
            scheduler.last_epoch=-1

        # train the model
        model.train( ) # Set model to training mode
        loss, acc = _train_loop(model,dataloader_dict['train'],\
                                criterion,optimizer,device)
        loss_dict['train'].append(loss)
        acc_dict['train'].append(acc)
        scheduler.step()
        t = time.time()
        print 'Training: Elapsed time: {:.4f} Loss: {:.4f} Acc: {:.4f}'.\
                format(t - t_start, loss, acc)

        if use_val == False and acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_prog:
                torch.save(model,prog_fname)

        # Validation is only entered when use_val is True
        if use_val:
            model.eval() # Set model to evaluate mode
            loss, acc  = _val_loop(model,dataloader_dict['val'],\
                                   criterion,device)
            loss_dict['val'].append(loss)
            acc_dict['val'].append(acc)

            t = time.time()
            print 'Validation: Elapsed time: {:.4f} Loss: {:.4f} Acc: {:.4f}'.\
                format(t - t_start, loss, acc)

            if  acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_prog:
                    torch.save(model,prog_fname)

    t_final = time.time()
    print 'Total time {:.4f}'.format(t_final - t_start)

    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict

def imshow_tensor(image_tensor):
    '''
    Plot the image tensors from the training set.
    If a Dataloader is used, use torchvision.utils.make_grid(image_tensor) first.
    '''
    arr = image_tensor.numpy().transpose(1,2,0)
    arr = STD*arr + MEAN
    arr = np.clip(arr,0,1)
    plt.imshow(arr)
