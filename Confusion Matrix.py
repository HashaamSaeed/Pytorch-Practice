import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix    ## creats ndarray confusion matrix 

import itertools
import numpy as np



torch.set_printoptions(linewidth=120)

#torch.set_grad_enabled(True)


train_set= torchvision.datasets.FashionMNIST(         ## data set instance is named as train_set 
	root='./data'
	,train=True
	,download=True
	,transform=transforms.Compose([
		transforms.ToTensor()            ## this transforms data into a tensor
		])
	)


class Network(nn.Module):
	def __init__(self):
		super(Network,self).__init__()
		self.conv1= nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)  ## in channels for the first layer input depend on the colour channel
		self.conv2= nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)  # stride is set to 1,1 by default, out_channel refers to No.Of filters used
		self.fc1= nn.Linear(in_features=12*4*4,out_features=120)           ## As a rule of thumb output of last layer is input of first layer thats why                                                                    # out is in fc1 120 and in is 120 in fc2, also bias is true by default
		self.fc2= nn.Linear(in_features=120,out_features=60)
		self.out= nn.Linear(in_features=60,out_features=10)               ## out feature of the last layer depend on the No.of classes to identify by the network
		                                                                   # for FMNIST its 1o cuz 10 classes
	def forward(self,t):
		## (1) implement forward pass with input t
		t = t
		## (2) hidden conv layer
		t = self.conv1(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)  ## this is just after the conv being done think of this as a form
		                                               # of reduction , it does not have any weights
		## (3) hidden conv layer
		t = self.conv2(t)
		t = F.relu(t)
		t = F.max_pool2d(t, kernel_size=2, stride=2)
		## (4) hidden linear layer
		t = t.reshape(-1, 12 * 4 * 4)          ## When passing from conv layer to linear layer we flatten the tensor
		                                        # -1 refers to auto channel identification, 4*4 are the output dimensions 
		                                        # of the 12 output convulutons in the previous layer. from 28x28 to 4x4 after
		                                        # passing through 2 Conv channels
		t = self.fc1(t)
		t = F.relu(t)


		## (5) hidden linear layer
		t = self.fc2(t)
		t = F.relu(t)

		# (6) output layer
		t = self.out(t)
		#t = F.softmax(t, dim=1)     ## usualy RelU is used for hidden layer but softmax is used in the output because
		                              # we need to predict single category for the 10 classes therefore softmax returns
		                              # a positive predicition probability for each class that adds to 1 in total
		                              # However, in our case, we won't use softmax() because the loss function that we'll
		                              # use, F.cross_entropy(), implicitly performs the softmax() operation on its input, 
		                              # so we'll just return the result of the last linear transformation
		return t		

network = Network()

def get_num_correct(preds,labels):          
	return preds.argmax(dim=1).eq(labels).sum().item()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000)
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(1):
    
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader: # Get Batch
        images, labels = batch 

        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch", epoch, 
        "total_correct:", total_correct, 
        "loss:", total_loss
    )


## to make the confusion matrix we need to collect all values 
#@torch.no_grad()   ## another way to turn off grad track locally
def get_all_preds(model, loader):
    all_preds = torch.tensor([])  ## empty torch tensor 
    for batch in loader:   ## go through all of our batches in data loader
        images, labels = batch

        preds = model(images)   ## passing images to the model
        all_preds = torch.cat(   ## concact the prediction in the all preds tensor at dim =0
                                   # this gives us a tensor with all the predictions
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

with torch.no_grad():      ##Python's with context manger keyword to specify that a specify block of code should exclude gradient computations
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

## extra stuff 
# print(train_preds.shape)    
# print(train_preds.requires_grad)  ## will retur true by default if above no grad is used in data loader
# print(train_preds.grad)            # wont return any val since grad is not tracking                                    

preds_correct = get_num_correct(train_preds, train_set.targets)
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))


## building the cofusion matrix 

stacked = torch.stack(              ## stacking func to concatonate the two tensors , gives tuple [9,8] orig val vs pred val
    (                               ## stacked.shape = [60000,2]
        train_set.targets       ## labels from the data set
        ,train_preds.argmax(dim=1)  ## predicted labels in the pred tensor
    )
    ,dim=1                            ## dimension where we need to insert     
)

stacked[0].tolist()  ## conversts a tensor into an int list for python gives output [9,9]

cmt = torch.zeros(10,10, dtype=torch.int32)    ## cmt initialising


## gives the confusioon tensor of [10,10] like in the pic we saw of comparision
for p in stacked:
    tl, pl = p.tolist()       ## does unpacking of variables again , allocates t1 to first val in tuple and p1 to second val in tuple if stacked [9,9] then both will be 9
    cmt[tl, pl] = cmt[tl, pl] + 1  ## find val in cmt to put and add one

print(cmt)    

cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))  ## returns an nDarray of numpy for confusion matrix witghout doing all of the shit above
print(type(cm))
print(cm)


## plotting the confusion matrix 
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

names = (
    'T-shirt/top'
    ,'Trouser'
    ,'Pullover'
    ,'Dress'
    ,'Coat'
    ,'Sandal'
    ,'Shirt'
    ,'Sneaker'
    ,'Bag'
    ,'Ankle boot'
)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names)
