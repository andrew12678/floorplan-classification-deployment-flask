from torchvision import models
import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *input):
        """

        :param input:
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        :return:
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class ResNet18Model(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        _layers = list(backbone.children())
        last_layer_in = _layers[-1].in_features
        self.feature_extractor = nn.Sequential(*_layers[:-1])
        self.clf = nn.Linear(last_layer_in, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        # work out why this is the case-->Linear expecteds (N, Features) but we have (N,F,1,1)
        x = x.squeeze(-1).squeeze(-1)
        # These are both synonyms
        #Linear
        #x = torch.flatten(x, 1)
        x = self.clf(x)
        return x

if __name__ == "__main__":
    model = ResNet18Model(2)
    print(model)