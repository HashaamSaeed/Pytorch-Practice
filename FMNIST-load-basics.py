import torch           ## tensor lib
import torchvision     ## vision lib
import torchvision.transforms as transforms   ## common transform for img

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)   ## for console


train_set= torchvision.datasets.FashionMNIST(         ## data set instance is named as train_set 
	root='./data'
	,train=True
	,download=True
	,transform=transforms.Compose([
		transforms.ToTensor()            ## this transforms data into a tensor
		])
	)

train_loader = torch.utils.data.DataLoader(train_set       ## training_set was loaded onto the data loader class
                                          ,batch_size=10  # this gives us to access/query data when using it like
                                          ,shuffle=True)    # shuffle, batch size etc
print(len(train_set))          ## tells length of img in data
print(train_set.targets)       ## label tensor of the classes for tshirt etc
print(train_set.targets.bincount()) ## tells how many of each label/class exists in the dataset
sample = next(iter(train_set))  ## the iter func in python returns an object representing a stream of data  
                                 # the next func gets the next objkect in the stream                                                      
print(len(sample))
print(type(sample))
image, label = sample            ## sequence unpacking allows us to acces data objects, its  a short form for 
                                  # image = sample[0],label=sample[1]
print('image:',image.shape)
#label.shape                      ## this has now become an int therefore no longer a tensor 
#plt.imshow(image.squeeze(),cmap='gray')
#plt.show()
print('label:',label)

batch= next(iter(train_loader))
print('batch',len(batch),type(batch))
images,labels = batch                 ## here it is images, labels cuz more are used 
print(images.shape,labels.shape)      ## here label is a rank 1 tensor with axis length as 10(batch size) 


grid = torchvision.utils.make_grid(images, nrow=10) ## grid func for tensors,nrow specifies no. of img in a 
                                                     # row to display the batch of img tensor
plt.figure(figsize=(10,10))                          ## figure box size
plt.imshow(np.transpose(grid, (1,2,0)))              ## For a colored image plt.imshow takes image dimension in 
                                                      # following form [height width channels] ...while pytorch 
                                                      # follows [channels height width]... so for compatibility 
                                                      # we have to change pytorch dimensions so that channels 
                                                      # appear at end... the standard representation of array 
                                                      # is [axis0 axis1 axis2].... so we have to convert (0,1,2) 
                                                      # to (1,2,0) form to make it compatible for imshow
plt.show()
print('labels:', labels)





#Alternate way to show images for the batch
#############################################################################################
#how_many_to_plot = 20
#train_loader = torch.utils.data.DataLoader(
#    train_set, batch_size=1, shuffle=True
#)
#mapping = {
#    0:'Top', 1:'Trousers', 2:'Pullover', 3:'Dress', 4:'Coat'
#   ,5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle Boot'
#}
#plt.figure(figsize=(50,50))
#for i, batch in enumerate(train_loader, start=1):
#    image, label = batch
#    plt.subplot(10,10,i)
#    fig = plt.imshow(image.reshape(28,28), cmap='gray')
#    fig.axes.get_xaxis().set_visible(False)
#    fig.axes.get_yaxis().set_visible(False)
#    plt.title(mapping[label.item()], fontsize=28)
#    if (i >= how_many_to_plot): break
#plt.show()
########################################################################################################