import torch as th
import torch.nn.functional as F 


class StoneLoss(th.nn.Module):
    def __init__(self, weight=1.0, num_classes=3):
        super(StoneLoss, self).__init__()
        self.weight = weight
        self.loss_fn = th.nn.CrossEntropyLoss()
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        
        #predictions = th.softmax(predictions, dim=-1)  
        targets_one_hot = F.one_hot(targets.to(th.long), num_classes=self.num_classes) 
        
        #t90 = th.rot90(targets, k=1, dims=[1,2])
        #t180 = th.rot90(targets, k=2, dims=[1,2])
        #t270 = th.rot90(targets, k=3, dims=[1,2])

        loss0 = self.loss_fn(predictions, targets_one_hot.float())
        #loss90 = self.loss_fn(predictions, t90.float())
        #loss180 = self.loss_fn(predictions, t180.float())
        #loss270 = self.loss_fn(predictions, t270.float())
        #loss = th.min(th.stack([loss0, loss90, loss180, loss270]), dim=0).values 
        
        loss = loss0 
        return self.weight * loss
    
    
if __name__ == "__main__":
    loss = StoneLoss()
    preds = th.randn((2, 19, 19, 4))
    targets = th.randint(0, 4, (2, 19, 19))
    l = loss(preds, targets)
    print(l)