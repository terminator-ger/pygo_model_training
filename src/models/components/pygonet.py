import timm
from torch import nn
from src.models.yolo.models.yolo import YOLO

class PyGoNet(nn.Module):
    def __init__(self, name="mobilenetv3_mh_large_150d", pretrained=True, num_classes=10):
        super().__init__()
        if "mobilenet" in name:
            self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        elif "yolo" in name:
            self.anchor_boxes = [
                                # Small anchor boxes
                                [[10,13], [16,30], [33,23]],
                                # Medium anchor boxes
                                [[30,61], [62,45], [59,119]],
                                # Large anchor boxes
                                [[116,90], [156,198], [373,326]]
                                ]
            self.net = YOLO(num_class=num_classes, anchor_boxes=self.anchor_boxes)

    def forward(self, x):
        return self.net(x)
    

if __name__ == "__main__":
    model = PyGoNet()
    print(model)