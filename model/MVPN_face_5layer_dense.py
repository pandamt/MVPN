# 5-layer dense MVPN - cope1(face) prediction
import sys, os, time
import nibabel as nib
import numpy as np
import itertools as it
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import face_dataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from scipy.stats.stats import pearsonr
from model_settings import total_run, sub, num_epochs, save_freq, print_freq, batch_size, learning_rate

sys.argv = [sys.argv[0], sys.argv[1]]
this_run = int(sys.argv[1])
print("this_run:", this_run)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize parameters
save_dir = '/gsfs0/data/fangmg/MVPD/model/model_runs/MVPD_net_' + sub + '/' + sub + '_cope1_face_5layer100_dense/MVPD_net_' + sub + '_testrun' + str(this_run) + '_epoch' + str(num_epochs) 
# create output folder if not exists
if not os.path.exists(save_dir):
       os.mkdir(save_dir)

# Hyper-parameters of NN 
input_size = 240 # FFA,STS,OFA each 80 voxels
hidden_size = 100 
output_size = 53539 # number of non-zero voxels in the brainmask 

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_size) 
        self.fc1 = nn.Linear(input_size, hidden_size) 
        #self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(input_size+hidden_size)
        self.fc2 = nn.Linear(input_size+hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(input_size+hidden_size*2)
        self.fc3 = nn.Linear(input_size+hidden_size*2, hidden_size)
        self.bn4 = nn.BatchNorm1d(input_size+hidden_size*3)
        self.fc4 = nn.Linear(input_size+hidden_size*3, hidden_size)
        self.bn5 = nn.BatchNorm1d(input_size+hidden_size*4)
        self.fc5 = nn.Linear(input_size+hidden_size*4, hidden_size)
        self.bn6 = nn.BatchNorm1d(input_size+hidden_size*5)
        self.fc6 = nn.Linear(input_size+hidden_size*5, output_size)   
 
    def forward(self, x):
        out1 = self.fc1(self.bn1(x))
        #out = self.relu(out)
        in2 = torch.cat((x, out1), dim=1)
        out2 = self.fc2(self.bn2(in2))
        in3 = torch.cat((in2, out2), dim=1)
        out3 = self.fc3(self.bn3(in3))
        in4 = torch.cat((in3, out3), dim=1)
        out4 = self.fc4(self.bn4(in4))
        in5 = torch.cat((in4, out4), dim=1)
        out5 = self.fc5(self.bn5(in5))
        in6 = torch.cat((in5, out5), dim=1) 
        out = self.fc6(self.bn6(in6)) 
        return out

def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
       torch.nn.init.uniform_(m.weight, a=-0.5, b=0.5) 
       m.bias.data.fill_(0.01) 
       print(m.weight)

def train(net, trainloader, optimizer, epoch, print_freq, save_freq, save_dir):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader): 
        # get the inputs
        ROIs = data['ROI']
        GMs = data['GM']

        ROIs, GMs = Variable(ROIs.cuda()), Variable(GMs.cuda())
        #ROIs, GMs = Variable(ROIs), Variable(GMs) 
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(ROIs)
        #print("OUTPUTs:", outputs[0:10])
        loss = criterion(outputs, GMs)
        loss.backward()
        optimizer.step()
        # print statistics
        loss.item() 
        running_loss += loss.data
        #print("ROIs:", ROIs[0:10])
        if i % print_freq == (print_freq-1):    # print every print_freq mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i + 1, running_loss / print_freq))
            running_loss = 0.0
    if epoch % save_freq == 0: 
        save_model(net, optimizer, os.path.join(save_dir, 'MVPDnet_%03d.ckpt' % epoch))
        print("Model saved in file: " + save_dir + "/MVPDnet_%03d.ckpt" % epoch)

def test(net, test_loader, epoch, save_dir):
    net.eval()
    score = []
    GM_pred = []
    GM_target = []
    GM_pred = np.reshape(GM_pred, [-1, output_size])
    GM_target = np.reshape(GM_target, [-1, output_size])

    for i, data in enumerate(testloader):
            # get the inputs
            ROIs = data['ROI']
            GMs = data['GM']
            # wrap them in Variable
            ROIs, GMs = Variable(ROIs.cuda()),  Variable(GMs.cuda())
            #ROIs, GMs = Variable(ROIs),  Variable(GMs)
            # forward + backward + optimize
            outputs = net(ROIs)
            outputs_numpy = outputs.cpu().data.numpy()
            GM_pred = np.concatenate([GM_pred, outputs_numpy], 0)
            GMs_numpy = GMs.cpu().data.numpy()
            GM_target = np.concatenate([GM_target, GMs_numpy], 0)
            error = np.abs(outputs_numpy - GMs_numpy)
            min_error = np.min(error, 1)
            #print("idx:", i)
            #print("min_error:", min_error)
            #print("ROIs:", ROIs[0:10])
    # Variance explained
    GM_std = np.std(GM_target, axis=0)
    error_std = np.std(GM_target - GM_pred, axis=0)
    vari = np.zeros(output_size)
    # GM_r: Pearson's correlation coefficient; GM_p: 2-tailed p-value
    GM_r = [pearsonr(GM_pred[:, i], GM_target[:, i])[0] for i in range(output_size)]
    GM_p = [pearsonr(GM_pred[:, i], GM_target[:, i])[1] for i in range(output_size)]

    for i in range(output_size):
        if GM_std[i] != 0:
            vari[i] = 1 - np.square(error_std[i]/GM_std[i])
        if vari[i] < 0:
            vari[i] = 0

    max_vari = np.max(vari)
    print("max_vari:", max_vari)
    #print("vari:", vari[0:20])
    np.save(save_dir + '/variance_explained_%depochs.npy' % epoch, vari)
    np.save(save_dir + '/GM_pred_%depochs.npy' % epoch, GM_pred)
    np.save(save_dir + '/GM_target_%depochs.npy' % epoch, GM_target)
    np.save(save_dir + '/GM_r_%depochs.npy' % epoch, GM_r)
    np.save(save_dir + '/GM_p_%depochs.npy' % epoch, GM_p)

def save_model(net,optim,ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

if __name__ == "__main__":
    # Training 
    face_train = face_dataset.Face_Dataset()
    face_train.get_train(this_run, total_run)
    trainloader = DataLoader(face_train, batch_size, shuffle=True, num_workers=0, pin_memory=True) 
    # Testing 
    face_test = face_dataset.Face_Dataset()
    face_test.get_test(this_run, total_run)
    testloader = DataLoader(face_test, batch_size, shuffle=False, num_workers=0, pin_memory=True) 
   
    net = NeuralNet(input_size, hidden_size, output_size).to(device)
    #net.apply(init_weights)
    
    # Loss and optimizer
    criterion = nn.MSELoss() # mean squared error
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs+1):  # loop over the dataset multiple times
        train(net, trainloader, optimizer, epoch, print_freq, save_freq, save_dir)
        if (epoch != 0) & (epoch % save_freq == 0):
            test(net, testloader, epoch, save_dir)
