import torch 


## Concatenating joins a sequence of tensors along an existing axis, and stacking joins a sequence of 
 # tensors along a new axis.

t1 = torch.tensor([1,1,1])   ## single axis length 3 , torch.Size([3])
print(t1.unsqueeze(dim=0))   ## increasing axis to 2 at axis 0, output = tensor([[1, 1, 1]]), torch.Size([1, 3])
print(t1.unsqueeze(dim=1))   ## increasing axis to 2 at axis 1, output =tensor([[1], torch.Size([3, 1])
                              #                                       [1],
                              #                                       [1]]) 

t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])                              

torch.cat((t1,t2,t3),dim=0)    ## output = tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
torch.stack((t1,t2,t3),dim=0)  ## output = tensor([[1, 1, 1],
                                #                  [2, 2, 2],
                                #                  [3, 3, 3]])


torch.cat((t1.unsqueeze(0),t2.unsqueeze(0),t3.unsqueeze(0)),dim=0)  ## output = tensor([[1, 1, 1],
                                                                     #                  [2, 2, 2],
                                                                     #                  [3, 3, 3]])

torch.stack((t1,t2,t3),dim=1)       ## output = tensor([[1, 2, 3],
                                     #                  [1, 2, 3],
                                     #                  [1, 2, 3]])    

torch.cat((t1.unsqueeze(1),t2.unsqueeze(1),t3.unsqueeze(1)),dim=1)   ## output =  tensor([[1, 2, 3],
                                                                      #                   [1, 2, 3],
                                                                      #                   [1, 2, 3]])
## Concatenating joins a sequence of tensors along an existing axis thats why you see it ikke that above
# we cannot concat this sequence of tensors along the second axis because there currently is no second axis in existence, so in this case, stacking is our only option.

## understanding above concept
t1.unsqueeze(1)  ##  tensor([[1],
                  #          [1],
                  #          [1]])


t2.unsqueeze(1)    ## tensor([[2],
                      #         [2],
                      #         [2]])

        
## t3 is the same like above so you can understand what happened ...

## to join with stack or cat we need the tensors to be of matching shape
# therefore assunuing you have a rank 4 tensor of images 
# and you need to add images to it of rank3 tensor since they lack the batch axis
# you'll add them by first stacking the 3 images to add a new batch axis and then cat them
# with the other 4 axis tensor                                                                                                                                                   