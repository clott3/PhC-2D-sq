
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim

from PhC2d_data import PhCdata # Dataset class
from PhC2d_model import PhCNet # model class

def reset_seeds(seed):
    '''This resets all the random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

def calfloss(output,target):
    'Defines fractional loss of net output with data'
    loss=torch.abs(output - target)
    loss=torch.div(loss,target)
    loss[loss==float("Inf")] = 0 #ignore divide by 0 errors. this occurs at gamma point
    loss[torch.isnan(loss)]=0 #ignore nan errors
    loss=torch.mean(loss)
    return loss

def weights_init(m):
    'custom weights initialization'
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_and_eval_model(args, model, fv, ks, train_dataloader, valid_dataloader, test_dataloader, device):
    ## Define loss criteria, optimizer and adaptive learning scheduler
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      patience=1,
                                                      threshold=0.01,
                                                      verbose=True)
    writer = SummaryWriter("runs/"+f"BS={args.batchsize}_maxEp={args.maxepoch}_LR={args.learning_rate}_ks={ks}_model={fv}")

    for epoch in range(1,args.maxepoch+1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)

            optimizer.zero_grad() # zero the parameter gradients
            loss = criterion(model(x_batch), y_batch)
            loss.backward() # backpropagate loss
            optimizer.step() # update parameters
            running_loss += loss.item()/args.nbands

        print("Epoch = "+str(epoch)+"  :Total loss = "+str(running_loss))
        scheduler.step(running_loss) #this adjusts the adaptive LR scheduler
        writer.add_scalar('Loss', running_loss, epoch) # log training stats

    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(valid_dataloader):
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            pred_batch = model(x_batch)
            loss = criterion(pred_batch, y_batch)
            tloss = loss.item()

            floss = calfloss(pred_batch,y_batch)
            ploss = floss.item()

        writer.add_scalar('valid loss',tloss)
        writer.add_scalar('valid fractional loss',ploss)
        print('Total valid loss is '+str(ploss))

        for step, batch in enumerate(test_dataloader):
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            pred_batch = model(x_batch)
            loss = criterion(pred_batch, y_batch)
            tloss = loss.item()

            floss = calfloss(pred_batch,y_batch)
            ploss = floss.item()

        writer.add_scalar('test loss',tloss)
        writer.add_scalar('test fractional loss',ploss)
        print('Total test loss is '+str(ploss))

    writer.close()

def main(args):

    reset_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # we use GPU 0 by default
    print(device)

    # load h5 file for data
    if args.pol == 'TM':
        h5file = args.path_to_h5+'/sqTM-res64.h5'
    elif args.pol == 'TE':
        h5file = args.path_to_h5+'/sqTM-res64.h5'
    else:
        raise ValueError("Polarization can only be TM or TM")

    nsam = args.nsam # no. of samples
    input_size = 32 # downsize to 32x32 (default dataset is 64x64)
    nbands = args.nbands # no. of bands to predict

    # Define train-val-test split
    ntrain = int(0.7*nsam)
    nvalid = int(0.15*nsam)
    ntest = nsam - ntrain - nvalid

    # Define dataset and dataloaders
    train_dataset = PhCdata(h5file, ntrain, nvalid, ntest,
                            split = 'train', nbands = nbands, input_size = input_size)
    valid_dataset = PhCdata(h5file, ntrain, nvalid, ntest,
                            split = 'valid', nbands = nbands, input_size = input_size)
    test_dataset = PhCdata(h5file, ntrain, nvalid, ntest,
                            split = 'test', nbands = nbands, input_size = input_size)

    train_dataloader = data.DataLoader(train_dataset, batch_size = args.batchsize, shuffle=True) # Define dataloader, shuffle every epoch
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=len(valid_dataset)) # one batch
    test_dataloader = data.DataLoader(test_dataset, batch_size=len(test_dataset)) # one batch

    # Define network architecture, i.e. channel depths (conv) and no. of nodes (fc). Defined here to ease modification of architecture
    fv = [4,32,128,256,64,256,512,512,512]
    ks = 11 # define kernel size of conv layers

    net = PhCNet(fv,ks,nbands).to(device) # initialize network
    net.apply(weights_init) #apply custom weight initialization

    ## TRAIN AND TEST MODEL##
    train_and_eval_model(args,net,fv,ks,train_dataloader,valid_dataloader,test_dataloader,device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_h5', type = str, help = 'path to directory with h5 data', default = '/media/charlotte/DATA/')
    parser.add_argument('--pol', type = str, help = 'TM or TE', default = 'TM')
    parser.add_argument('--nbands', type=int, help='num of bands to predict',default=6)
    parser.add_argument('--nsam', type=int, help='num of training samples',default=20000)

    # Hyperparameters
    parser.add_argument('--learning_rate',type=float, help='pretraining/maml learning rate', default=1e-4)
    parser.add_argument('--batchsize',type=int, help='batchsize',default=32)
    parser.add_argument('--maxepoch',type=int, help='total no. of epochs to train', default=40)
    parser.add_argument('--seed',type=int, help='random seed', default=1)

    args = parser.parse_args()

    main(args)
