import torch
import time
import h5py
import numpy as np
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter
import math

writer = SummaryWriter('runs/')

starttime=time.time()

def reset_seeds():
    '''This resets all the random seeds'''
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

reset_seeds()
# sets device to GPU if avaiable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(device)

#load training data, by default resolution 64 x 64
f = h5py.File('sqTM-res64.h5','r')

nsam=20000 #no. of samples
isize=32 #downsize to 32x32 

x_train=[]
y_train=[],[],[],[],[],[] # no. of bands to predict

nbands = 6 # number of bands to predict

print("Retrieving data..")

### The following code retrieves the input unit cell and the 
### eigenfrequencies (bands) from the h5 file containing the band data
### Access to this dataset can be requested via email 

for memb in np.arange(nsam):
    eps=f.get('shapes/'+str(memb+1)+'/unitcell/epsilon')
    epslow=f.get('shapes/'+str(memb+1)+'/unitcell/epsilon_comput')
    efreq=f.get('shapes/'+str(memb+1)+'/eigfreqs')
    epsimage=np.array(epslow)
    epsresize=resize(epsimage,(isize,isize))
    x_train.append(epsresize)
    for i in range(nbands):
        y_train[i].append(np.ravel((efreq[i])))
print("Data Retrieved!")

#rescale eps values to between [0,1] from [1,10]
x_data = (np.array(x_train).astype('float32')-1) / 9
y_data = np.array(y_train).astype('float32')

#Split data into train:valid:test in 0.7:0.15:0.15 ratio
nvalid=int(0.15*nsam)
ntest=int(0.15*nsam)
ntrain=int(0.7*nsam)
nfrom=ntest+nvalid

# Initalize the eigenfrequencies. Eigenfrequencies are computed at 
# 23*23=529 points per band
y_train=np.zeros((nbands,ntrain,529))
y_valid=np.zeros((nbands,nvalid,529))
y_test=np.zeros((nbands,ntest,529))


#Random shuffle the ordering of dataset (only done once)
rp = np.random.permutation(len(x_data))
x_data = x_data[rp]
(x_train,x_valid,x_test)=x_data[nfrom:], x_data[ntest:nfrom], x_data[:ntest]

for i in range(nbands):
    assert len(y_data[i]) == len(x_data)
    y_data[i] = y_data[i][rp]
    (y_train[i],y_valid[i],y_test[i]) = (y_data[i][nfrom:],
                                         y_data[i][ntest:nfrom], 
                                         y_data[i][:ntest])


#reshape inpute from (32,32) to (1,32,32)
w, h = isize, isize
x_train = x_train.reshape(x_train.shape[0],1,w,h)
x_valid = x_valid.reshape(x_valid.shape[0],1,w,h)
x_test = x_test.reshape(x_test.shape[0],1,w,h)

#convert to torch tensors and push to GPU
x_train_t=torch.from_numpy(x_train).to(device)
x_valid_t=torch.from_numpy(x_valid).to(device)
x_test_t=torch.from_numpy(x_test).to(device)

y_train_t=torch.from_numpy(y_train).to(device)
y_valid_t=torch.from_numpy(y_valid).to(device)
y_test_t=torch.from_numpy(y_test).to(device)

datatime = time.time()
print("data retrieval time =", datatime-starttime)
## DEFINE AND BUILD MODEL ##

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def calfloss(output,target):
    'Defines fractional loss of net output with data'
    loss=torch.abs(output - target)
    loss=torch.div(loss,target)
    loss[loss==float("Inf")] = 0 #ignore divide by 0 errors
    loss=torch.mean(loss)
    return loss

def unison_shuffle(x,y):
    'Shuffles arrays x and y in unison'
    yout = torch.zeros(y.shape,device=device)
    p = torch.randperm(len(x))
    if len(y) <= 6:
        for i in range(len(y)):
            assert len(y[i]) == len(x)
            yout[i] = y[i][p]
    else:
        # if model only contains one band
        assert len(x) == len(y)
        yout = y[p]
    return x[p], yout

def weights_init(m):
    'custom weights initialization'
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Define network architecture, i.e. channel depths (conv) and no. of nodes (fc) 
# Defined here to ease modification of architecture
fv = [4,32,128,256,64,256,512,512,512]
# define kernel size of conv layers
ks = 11

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.enc_block = nn.Sequential(
            nn.Conv2d(1, fv[0], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(fv[0]),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(fv[0], fv[1], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(fv[1]),
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(fv[1], fv[2], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
            nn.BatchNorm2d(fv[2]),
            nn.ReLU(),
            nn.MaxPool2d(4,4)
            )
        self.enc_linear = nn.Sequential(
            nn.Linear(fv[2] * 1 * 1, fv[3]),
            nn.ReLU(),
            nn.Linear(fv[3], fv[4]),
            nn.ReLU()
            )

        self.firstband = self.fc_block()
        self.secondband = self.fc_block()
        self.thirdband = self.fc_block()
        self.fourthband = self.fc_block()
        self.fifthband = self.fc_block()
        self.sixthband = self.fc_block()
        
    def fc_block(self):
        return nn.Sequential(
            nn.Linear(fv[4], fv[5]),
            nn.ReLU(),
            nn.Linear(fv[5], fv[6]),
            nn.ReLU(),
            nn.Linear(fv[6], fv[7]),
            nn.ReLU(),
            nn.Linear(fv[8], 529)
            )

    def forward(self, x):
        x = self.enc_block(x)
        x = x.view(-1, fv[2] * 1 * 1) # flatten
        x = self.enc_linear(x)
        
        b0 = self.firstband(x)
        b1 = self.secondband(x)
        b2 = self.thirdband(x)
        b3 = self.fourthband(x)
        b4 = self.fifthband(x)
        b5 = self.sixthband(x)
        
        return b0, b1, b2, b3, b4, b5
    
net1=Net()  
print(net1)

# Define hyperparameters
learnrate=0.0001
batchsize=64
maxepoch=40
savestring ='s6b_TM'
        
comment=savestring +f"_BS={batchsize} maxEp={maxepoch} \
    LR={learnrate} ks={ks} model={fv}"
writer = SummaryWriter(comment=comment)

reset_seeds()
net=Net()

net.to(device)
net.apply(weights_init)

## Define loss criteria, optimizer and adaptive learning scheduler
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.RMSprop(net.parameters(), lr=learnrate, alpha=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                  patience=1,
                                                  threshold=0.01,
                                                  verbose=True)
## BEGIN TRAINING ##

starttrain = time.time()  
print('Now Training..')

epoch=0
lossvec = []
while epoch < maxepoch:
    epoch += 1
    nbatch=0
    running_loss = 0.0
    # one epoch is going through whole training set once
    niter=len(y_train_t[0])//batchsize 
    # for every new epoch, random shuffle training samples order
    x_train_t, y_train_t = unison_shuffle(x_train_t,y_train_t) 
   
    for _ in range(niter):
        
        x_batch = x_train_t[batchsize*nbatch:batchsize*(nbatch+1)]
        y0_batch = y_train_t[0][batchsize*nbatch:batchsize*(nbatch+1)]
        y1_batch = y_train_t[1][batchsize*nbatch:batchsize*(nbatch+1)]
        y2_batch = y_train_t[2][batchsize*nbatch:batchsize*(nbatch+1)]
        y3_batch = y_train_t[3][batchsize*nbatch:batchsize*(nbatch+1)]
        y4_batch = y_train_t[4][batchsize*nbatch:batchsize*(nbatch+1)]
        y5_batch = y_train_t[5][batchsize*nbatch:batchsize*(nbatch+1)]
        
        nbatch += 1
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output0, output1, output2, output3, output4, output5 = net(x_batch)
        
        loss0 = criterion(output0, y0_batch)
        loss1 = criterion(output1, y1_batch)
        loss2 = criterion(output2, y2_batch)
        loss3 = criterion(output3, y3_batch)
        loss4 = criterion(output4, y4_batch)
        loss5 = criterion(output5, y5_batch)
        
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()/nbands
        
    print("Epoch = "+str(epoch)+"  ;Total loss for epoch = "+str(running_loss))
    scheduler.step(running_loss) #this adjusts the adaptive LR scheduler
    writer.add_scalar('Loss', running_loss, epoch)
    
print('Finished Training')
endtrain = time.time()
print("training time = ", endtrain-starttrain)

## EVALUATE VALIDATION LOSS ##

tloss = 0.0
ploss =0.0
total = x_valid_t.shape[0]
net.eval() #make net on evaluation mode

with torch.no_grad():
    
    out0, out1, out2, out3, out4, out5 = net(x_valid_t)
    testloss0 = criterion(out0,y_valid_t[0])
    testloss1 = criterion(out1,y_valid_t[1])
    testloss2 = criterion(out2,y_valid_t[2])
    testloss3 = criterion(out3,y_valid_t[3])
    testloss4 = criterion(out4,y_valid_t[4])
    testloss5 = criterion(out5,y_valid_t[5])
    testloss = testloss0 + testloss1 + testloss2 + testloss3 + testloss4 + testloss5
    tloss += testloss.item()/nbands
    
    floss0 = calfloss(out0,y_valid_t[0])
    floss1 = calfloss(out1,y_valid_t[1])
    floss2 = calfloss(out2,y_valid_t[2])
    floss3 = calfloss(out3,y_valid_t[3])
    floss4 = calfloss(out4,y_valid_t[4])
    floss5 = calfloss(out5,y_valid_t[5])
    floss = floss0 + floss1 + floss2+floss3 + floss4 + floss5
    ploss += floss.item()/nbands

writer.add_scalar('valid loss',tloss)
writer.add_scalar('valid percent loss',ploss)

print('Total valid loss on '+str(total)+' validation samples is '+str(tloss))
print('Total valid loss on '+str(total)+' validation samples is '+str(ploss))

## EVALUATE TEST LOSS ##
tloss = 0.0
ploss =0.0
total = x_test_t.shape[0]

with torch.no_grad():
    out0, out1, out2, out3, out4, out5 = net(x_test_t)
    testloss0 = criterion(out0,y_test_t[0])
    testloss1 = criterion(out1,y_test_t[1])
    testloss2 = criterion(out2,y_test_t[2])
    testloss3 = criterion(out3,y_test_t[3])
    testloss4 = criterion(out4,y_test_t[4])
    testloss5 = criterion(out5,y_test_t[5])
    testloss = testloss0 + testloss1 + testloss2 + testloss3 + testloss4 + testloss5
    tloss += testloss.item()/nbands
    
    floss0 = calfloss(out0,y_test_t[0])
    floss1 = calfloss(out1,y_test_t[1])
    floss2 = calfloss(out2,y_test_t[2])
    floss3 = calfloss(out3,y_test_t[3])
    floss4 = calfloss(out4,y_test_t[4])
    floss5 = calfloss(out5,y_test_t[5])
    floss = floss0 + floss1 + floss2+floss3 + floss4 + floss5
    ploss += floss.item()/nbands
    
writer.add_scalar('Test loss',tloss)
writer.add_scalar('Test percent loss',ploss)

print('Total test loss on '+str(total)+' test samples is '+str(tloss))
print('Total test loss on '+str(total)+' test samples is '+str(ploss))
          
endtime=time.time()
print("Total Time taken = "+ str(endtime-starttime))


writer.close()
torch.cuda.empty_cache()














