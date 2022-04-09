import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)




# How networks are built in layers
################################################################################
#class Network:
#	def _init_(self):
#	self.layer=None      ## single dummy layer in constructor

#	def forward(self,t):
#		t= self.layer(t)   ## dummy implemnetation for forward prop
#		return t           ## dummy layer takes in a tensor t and return the 
		                    # transformed tensor

###############################################################################

## kernal size is the filter size
## out_channels	Sets the number of filters. One filter produces one output channel.
## out_features	Sets the size of the output tensor for linear layer for rank 1 tensors
## in_channels means one input 
## linear layers also called fully connected (FC layer) or (dense layer)

#conv1	in_channels	1	the number of color channels in the input image.
#conv1	kernel_size	5	a hyperparameter.
#conv1	out_channels	6	a hyperparameter.
#conv2	in_channels	6	the number of out_channels in previous layer.
#conv2	kernel_size	5	a hyperparameter.
#conv2	out_channels	12	a hyperparameter (higher than previous conv layer).
#fc1	in_features	12*4*4	the length of the flattened output from previous layer.
#fc1	out_features	120	a hyperparameter.
#fc2	in_features	120	the number of out_features of previous layer.
#fc2	out_features	60	a hyperparameter (lower than previous linear layer).
#out	in_features	60	the number of out_channels in previous layer.
#out	out_features	10	the number of prediction classes.





class Network(nn.Module):
	def __init__(self):
		super(Network,self).__init__()
		self.conv1= nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)  ## in channels for the first layer input depend on the colour channel
		self.conv2= nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)  # stride is set to 1,1 by default, out_channel refers to No.Of filters used
		
		self.fc1= nn.Linear(in_features=12*4*4,out_features=120)           ## As a rule of thumb output of last layer is input of first layer thats why
		                                                                    # out is in fc1 120 and in is 120 in fc2, also bias is true by default
		self.fc2= nn.Linear(in_features=120,out_features=60)
		self.out= nn.Linear(in_features=60,out_features=10)               ## out feature of the last layer depend on the No.of classes to identify by the network
		                                                                   # for FMNIST its 1o cuz 10 classes

def forward(self,t):
		## (1) implement forward pass with input t
		t= t

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

##Hyperparameters are choosen randomly like kernal size,out_channels etc  to be test and ttuned


network = Network()
print(network)     ## prints the network

print(network.conv1)  ## print individual layers 
print(network.conv1.weight,network.conv1.weight.shape) ## print layer weights and shape which are tensors

## for conv layer the weight lives in the filters 
print(network.conv1.weight[0].shape)      ## tells about the filters depth(No.of filters) and dimensions
## linear layers have a rank 2 tensor . out_ chan is the Height and in_chan is the width of the tensor

## simple Mat Mul
in_features= torch.tensor([1,2,3,4],dtype=torch.float32)
weight_matrix=torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=torch.float32)
weight_matrix.matmul(in_features)


## way to check all weight parameters in a network
for param in network.parameters():
	print(param.shape)

## way to check all weight parameters in a network with BIAS
for name, param in network.named_parameters():
	print(name,'\t\t',param.shape)

fc= nn.Linear(in_features=4,out_features=3,bias=False)  ## creating layer outside of main classn 
print(fc(in_features))    ## prints the weights 
fc.weight=nn.Parameter(weight_matrix)  ## inputing coustom weight instead of random
print(fc.weight)
print(fc(in_features)) ## passing the tensor in_features to the fc layer

## if input has 3 elements in the last Axis then we will have 3 inputs to the model
