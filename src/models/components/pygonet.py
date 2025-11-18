import timm
from torch import nn

class PyGoNet(nn.Module):
    def __init__(self, name="mobilenetv3_mh_large_150d", pretrained=True, num_classes=10):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":
    model = PyGoNet()
    print(model)