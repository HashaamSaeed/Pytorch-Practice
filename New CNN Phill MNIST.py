import torch as T 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
	def __init__(self,lr,epochs,batch_size,num_classes=10):
		super(CNN,self).__init__()
		self.epochs=epochs
		self.lr=lr
		self.batch_size=batch_size
		self.num_classes=num_classes
		self.loss_history=[]
		self.acc_history=[]
		self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu' )  

		self.conv1=nn.Conv2d(1,32,3)
		self.bn1=nn.BatchNorm2d(32,momentum=0.1)

		self.conv2= nn.Conv2d(32,32,3)
		self.bn2= nn.BatchNorm2d(32,momentum=0.1)

		self.conv3=nn.Conv2d(32,32,3)
		self.bn3=nn.BatchNorm2d(32,momentum=0.1)
		self.maxpool1=nn.MaxPool2d(2)

		self.conv4=nn.Conv2d(32,64,3)
		self.bn4=nn.BatchNorm2d(64,momentum=0.1)

		self.conv5=nn.Conv2d(64,64,3)
		self.bn5=nn.BatchNorm2d(64,momentum=0.1)

		self.conv6=nn.Conv2d(64,64,3)
		self.bn6=nn.BatchNorm2d(64,momentum=0.1)
		self.maxpool2=nn.MaxPool2d(2)  ## reduce factor by 2

		input_dims = self.calc_input_dims()    ## func late defined to calc dimensions for cnvs

		self.fc1= nn.Linear(input_dims,self.num_classes)  ## take in conv layers and then compute the class it belongs to 

		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)    ## self.parameters is inherited from the nn module and provides access to the parameters of the conv and dense layers

		self.loss= nn.CrossEntropyLoss()

		self.to(self.device)  ## sends the network to the prefered device like CPU or GPU

		self.get_data()  ## gettinf MNIST


	def calc_input_dims(self):
		batch_data = T.zeros((1,1,28,28))  ## 4D zero tensor
		
		batch_data= self.conv1(batch_data)  ## sends it to the first layer to calculate 
		# batch_data= self.bn1(batch_data)

		batch_data= self.conv2(batch_data)
		# batch_data= self.bn2(batch_data)

		batch_data= self.conv3(batch_data)
		# batch_data= self.bn3(batch_data)
		batch_data= self.maxpool1(batch_data)

		batch_data= self.conv4(batch_data)
		# batch_data= self.bn4(batch_data)

		batch_data= self.conv5(batch_data)
		# batch_data= self.bn5(batch_data)

		batch_data= self.conv6(batch_data)
		# batch_data= self.bn6(batch_data)
		batch_data= self.maxpool2(batch_data)

		return int(np.prod(batch_data.size()))  ## will give number of elements in batch size of 1



	def forward(self,batch_data):
		batch_data=T.as_tensor(batch_data).to(self.device)  ## lower case tensor preserves incoming data type T.Tensor doesnt, also this tensor passes it to the preferred device

		batch_data= self.conv1(batch_data)  ## sends it to the first layer to calculate 
		batch_data= self.bn1(batch_data)
		batch_data= F.relu(batch_data)
		


		batch_data= self.conv2(batch_data)
		batch_data= self.bn2(batch_data)
		batch_data= F.relu(batch_data)

		batch_data= self.conv3(batch_data)
		batch_data= self.bn3(batch_data)
		batch_data= F.relu(batch_data)
		batch_data= self.maxpool1(batch_data)

		batch_data= self.conv4(batch_data)
		batch_data= self.bn4(batch_data)
		batch_data= F.relu(batch_data)

		batch_data= self.conv5(batch_data)
		batch_data= self.bn5(batch_data)
		batch_data= F.relu(batch_data)

		batch_data= self.conv6(batch_data)
		batch_data= self.bn6(batch_data)
		batch_data= F.relu(batch_data)
		batch_data= self.maxpool2(batch_data)

		batch_data = batch_data.view(batch_data.size()[0],-1) ## flatten at the zeroth element by -1 getting a 2d array

		classes = self.fc1(batch_data)

		return classes


	def get_data(self):

		mnist_train_data= MNIST('mnist', train=True,download=True,transform=ToTensor())

		self.train_data_loader = T.utils.data.DataLoader(mnist_train_data,
			                                          batch_size=self.batch_size,
			                                          shuffle=True,
			                                          num_workers=8)
		
		mnist_test_data=MNIST('mnist',train=False,download=True,transform=ToTensor())

		self.test_data_loader=T.utils.data.DataLoader(mnist_test_data,
			                                           batch_size=self.batch_size,
			                                           shuffle=True,
			                                           num_workers=8)

	
	def _train(self):    ## nn.Module class already has train() func thats why we're using _train
		#self.train()                ## this func doesnt actually train it just tells pytorch that we're enteriong the 
		for i in range(self.epochs):  # trainging phase so update anything like grads, batch norms and etc but when in test phase dont update those metrics 
			ep_loss=0
			ep_acc=[]
			for j,(input,label) in enumerate(self.train_data_loader):  ## iterate over j and input,label tuple
				self.optimizer.zero_grad()   ## always do this
				label=label.to(self.device)   ## sending label to chosen device 
				prediction =self.forward(input)  ## feed forward pass for batch(input=image)
				loss = self.loss(prediction,label)  ## calc loss
				prediction=F.softmax(prediction,dim=1)  ## actual prediction of class by network on dim=1 sincce dim=0 is the image/batch
				classes =T.argmax(prediction,dim=1)  ## choosing the highest prob of label 
				wrong=T.where(classes != label,
					           T.tensor([1.]).to(self.device), ## if true gets 1
				               T.tensor([0.]).to(self.device)) ## if wrong gets 0
				acc= 1- T.sum(wrong)/self.batch_size ## calc for accuracy scaled to batch size

				ep_acc.append(acc.item())  ## accuracy of 1 epoch converted to a numpy array item value
				self.acc_history.append(acc.item()) ## just for keeping track of acc in 2 differetn places  
				ep_loss += loss.item()   ## epoch loss + epoch loss for the next epoch 
				loss.backward()
				self.optimizer.step()
			print('Finsih epoch',i,'total loss %.3f' % ep_loss,
				   'accuracy %.3f' % np.mean(ep_acc))
			self.loss_history.append(ep_loss)




	def _test(self):
		self.eval()                ## this func doesnt actually train it just tells pytorch that we're enteriong the 
		                           # trainging phase so dont update anything like grads, batch norms and etc 
		ep_loss=0
		ep_acc=[]
		for j,(input,label) in enumerate(self.test_data_loader):
			# self.optimizer.zero.grad()    ## no need here since test phase
			label=label.to(self.device)
			prediction =self.forward(input)
			loss = self.loss(prediction,label)
			prediction=F.softmax(prediction,dim=1)
			classes =T.argmax(prediction,dim=1)
			wrong=T.where(classes != label,
				           T.tensor([1.]).to(self.device),
			               T.tensor([0.]).to(self.device))
			acc= 1- T.sum(wrong)/self.batch_size

			ep_acc.append(acc.item())
			## self.acc_history.append(acc.item())  ## plotting not needed
			ep_loss += loss.item()
			# loss.backward()    ## not needed for test phase
			# self.optimizer.step()  ## not needed for test phase
		print('Finsih epoch test ','total loss %.3f' % ep_loss,
			   'accuracy %.3f' % np.mean(ep_acc))
		    # self.loss_history.append(ep_loss)  ## plotting not needed
		


 
if __name__ == '__main__':   ## understand this ----------
	network=CNN(lr=0.01,batch_size=1000,epochs=10)
	network._train()
	network._test()
	# plt.plot(network.loss_history)
	# plt.show()
	# plt.plot(network.acc_history)
	# plt.show()
	

		              