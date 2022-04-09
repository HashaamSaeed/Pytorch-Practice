import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


torch.set_printoptions(linewidth=120)

torch.set_grad_enabled(True)   ## turns gradient mapping off to save memory

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
	
                           
## IMP the numbers inside the filter kernal are the weights which are random!!!



## network = Network()


## for a single img not a batch
r"""
sample= next(iter(train_set))

image, label = sample
print(image.shape)

image.unsqueeze(0).shape   ## gives us batch of size 1

print(image.unsqueeze(0).shape)

pred = network(image.unsqueeze(0))  ##image shape needs to be (batch_size × in_channels × H × W)
print('pred',pred)   ## this outputs prediction for each img in the batch, they're not probs thats why later we sue softmax
print('pred shaoe',pred.shape)    ## returns [1,10] meaning we have 1 image in batch and 10 classes to represent its prob
print('pred label',label)         ## predicted label
print('pred argmax',pred.argmax(dim=1))  ## gives index of where the max val appears
print('pred softmax',F.softmax(pred,dim=1))   ## softmax the prediction into prob
print('softmax sum',F.softmax(pred,dim=1).sum())  ## always Prob sum to 1

"""
r"""
## for loading a batch to the 
data_loader = torch.utils.data.DataLoader(
	train_set
	,batch_size=100
	)

batch= next(iter(data_loader))

images,labels= batch

print('images',images.shape,'labels',labels.shape)

preds= network(images)
loss = F.cross_entropy(preds,labels)  ## calculating the loss
print('loss',loss.item())
print('grad no backprob',network.conv1.weight.grad)
loss.backward()      ## calculating gradients by backpropagation, without this above func will give "NONE"
print('grad',network.conv1.weight.grad,'grad shape',network.conv1.weight.grad.shape)

"""

r"""
print('preds',preds)
print('preds',preds.shape)
print('argmax',preds.argmax(dim=1))   ## index where highest val is present in prediction
print('preds labels',labels)
print(preds.argmax(dim=1).eq(labels))  ## equals function computes if label and argmax are same, means if prediction was right or wrong
print(preds.argmax(dim=1).eq(labels).sum()) ## tells how many predictions were right
"""
r"""
def get_num_correct(preds,labels):          ## function to give No.of of predictions that are correct as a python int not a tensor 
	return preds.argmax(dim=1).eq(labels).sum().item()

print(get_num_correct(preds,labels))

## updating the weights by SGD to minimise the loss fucntion

optimiser = optim.Adam(network.parameters(),lr=0.01)  ## network.parameters are networks weigths , lr sets the learning rate so be careful not to give too much
                                                       # to jump over the minimum
print('loss after optim',loss.item())    ## has no affect since we still ddint tell it to update grads
print('after optim',get_num_correct(preds,labels))
optimiser.step()           ## this updates the weight, meaning to step towards the minimum of the ADAM

preds=network(images)    ## passing img again after update is enables to check 
loss = F.cross_entropy(preds, labels)
print('after optim after update grad',loss.item())
print('after optim after update grad',get_num_correct(preds,labels))
"""

##Batch wise training in a shortened way ## training a single batch
r'''
############################################################################################################
def get_num_correct(preds,labels):         
	return preds.argmax(dim=1).eq(labels).sum().item()



network = Network()  ## Network instance 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

batch = next(iter(train_loader)) # Get Batch  for single batch = batch = next(iter(train_loader)) 
images, labels = batch

preds = network(images) # Pass Batch
loss = F.cross_entropy(preds, labels) # Calculate Loss

loss.backward() # Calculate Gradients
optimizer.step() # Update Weights

#-------------------------------------------
print('loss1:', loss.item())
preds = network(images)
loss = F.cross_entropy(preds, labels)
print('loss2:', loss.item())
'''


## training for whole batch 
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


