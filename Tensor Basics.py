import torch
import numpy as np

# print(torch.cuda.is_available())         ## tells if cuda is installed or not, for me its true
# print(torch.version.cuda)                ## gives cuda version   , i have installed ver 10.1

# t= torch.Tensor()           ## tensor created on CPU by default, # Capital T in tensor is when no data is entered
                              ## small t in tensor used when data is entered

## tensor created on GPU No.1 of the system!!!! Tensors need to be on the smae devide i.e CPU or GPU to perform
## computations
# t= t.cuda()
# print(t)

## how an array is made and accessed
# a= [1,2,3,4,5]
# print(a[2])           ## gives 3 as output no need for bullshit for loop'here it is'

# 2D array 
# dd= torch.tensor([  [1,2,3],
#                     [4,5,6],
#                     [7,8,9]  ])
# dd=dd.cuda()

# print(dd[1][2])                  ## gives 6 as output 

# s= torch.tensor(dd)              ## makes dd a tensor
# s=s.cuda()
# print('array s',s)
# print('type of s',type(s))       ## used for checking type
# print('shape of s',s.shape)      ## specifies shape

## IMP!! Size and shape of a tensor are same in pytorch
# s= dd.reshape(1,9)    # rehspaing does not affect the original tensor therefore you need to load it to another
                      # variable so that it can permanently take its new shape.
# s= s.cuda()                      
# print(s)
# print(type(s))
# print(s.shape ) 
# print('New dd',dd)

## tensor attributes in pytorch
# print(t.dtype)
# print(t.device)
# print(t.layout)

## creating torch tensore with numpy arrays 
# data= np.array([1,2,3])
# print(type(data))           ## output as numpy nd array
# s= torch.Tensor(data)       ## convert nd array to torch tensor 
# print(s,s.dtype)

# torch.Tensor(data)       ## out put as tensor float 32 , main constructor but lacks config like below 
# torch.tensor(data)       ## output as tensor variable type of the data entered in data var 32 , factory func
                           ## best used for everyday use
# torch.tensor(np.array([1,2,3]),dtype=torch.float64)    , sets var type even if int is passed  , factory func
# torch.as_tensor(data)    ## output as tensor variable type of the data entered in data var 32 , factory func
                           ## accepts any arr and is the best for performace cuz of low mem use
# torch.from_numpy(data)   ## output as tensor variable type of the data entered in data var 32 , factory func
                           ## only accepts numpy arr

## imp note is that the first 2 func above CANNOT adhere to changes in arrdata if you change it later on but the 
## last two WILL change so be careful about choosing the right one, thats cuz the first two copy data in memory
## but the last two share data in the memory 


##    creating tensor without data
# print(torch.eye(2))      #creates identity matrix of rank 2 and length 2
# print(torch.zeros([2,2]))
# print(torch.ones([2,2]))
# print(torch.rand([2,2]))

# t1 = torch.rand([3,4],dtype= torch.float32)
# t2 = torch.rand([3,4],dtype= torch.float32)
# t3 = torch.rand([3,4],dtype= torch.float32)
#print(torch.tensor(t.shape).prod())           ## convert shape into tensor and then prod which is 3x4 
#print(t,t.size(),len(t.shape),t.numel())      ## size and shape func are the same whereas the t.numel func gives 
                                               ## elements in the tensor and len func gives Axes/Rank, faster than
                                               # doing what we did above to get No. of elements
# print(t.reshape(6,2),t.reshape(2,2,3))       ## reshaping has to be product of the elements so can be any combo 
                                               #  like 6x2 , 12x1, 2x2x3 etc
# print(t.reshape(4,1,3).squeeze().shape)      ## removes dimensions that are 1                                 
# print(t.reshape(4,1,3).squeeze().unsqueeze(dim=0).shape)    ## Unsqueezing adds a dimension with a length of one.
                                  
# def flatten(t1):          ## declaration of function
#	t1= t1.reshape(1,-1)   ## the -1 tells the reshape() function to figure out what the value should
	                      # be based on the number of elements contained within the tensor  
	                      # therefore 1 x (det No. of elements).
#	t1= t1.squeeze()
#	return t1
# print(flatten(t1),flatten(t1).shape)

# print(torch.cat((t1,t2),dim=0))          ## Concatenating dim=0 means its axis wise addition of rows 
# print(torch.cat((t1,t2),dim=1))           # and dim=1 means its column wise addition of colum

#t1 = torch.rand([4,4],dtype= torch.float32)
#t2 = torch.rand([4,4],dtype= torch.float32)
#t3 = torch.rand([4,4],dtype= torch.float32)


#t= torch.stack((t1,t2,t3))                 ## another way to concatenate using stack but it does it by batch not 
                                            # specifically with rows or colums
#print(t,t.size())                          ## size tells us 3x4x4 using 4x4 tensors,therefore it means that
                                            # a batch of 3 with 3 Axis with 4 rows and 4 colm
 
#print(t.reshape(3,1,4,4),t.reshape(3,1,4,4).size())    ## Since CNN accept colour we added 1 more axis                                          
#print(t[0][0][0][0])                          ## accesing single elements in the stacked tensor not available 
                                                # since we didnt allocate the transformed tensor to some other 
                                                # variable                                 
#print(t.flatten(start_dim=1),t.flatten(start_dim=1).size()) ## smashes everything into a singel axis starting with
                                                             # the dimension mentione
#print(t1+t2)                                         ## element wise operation can only be done on similar 
                                                      # tensors with similar shapes like x,/,-,+
#print(t1.add(2))                                     ## this happens because pytorch converts the 2 into
                                                      # a tensor array matching the shaoe of t1 to add 
#print(np.broadcast_to(2,t1.shape))                   ## broadcasts shape of tensor t1 onto array of shape t1 with 2
#t4= torch.tensor([1,2,3,4],dtype=torch.float32)
#print(np.broadcast_to(t4.numpy(),t1.shape))   ## broadcasting shape only takes place row wise! it wont
                                               # add colums, so number of elements in first row should equal
#print(t1.le(1),t1.ge(2))                      ## various funcs le= less than, ge = greater than,done using broadcasting
#print(t1.abs(),t1.sqrt())

t= torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float32)
#print(t.sum(dim=0),t.prod(dim=1),t.mean(),t.std(),t.numel())   ## See how we reduced the No.of elements by using 
                                                                # func with dim as starting dimension/index
print(t.max(dim=0),t.argmax(dim=0))    ## gives max val in tensor, and argmax gioves index of max val 
                                        # in a flattened tensor if dim not mentioned dim tells the starting 
                                        # dimension
print(t.mean().item())                  ## gives output as a value not a tensor     
print(t.mean(dim=0).tolist())           ## gives output as a python list of array
print(t.mean(dim=0).numpy())            ## gives a numpy array            