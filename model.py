import torch
import torch.nn as nn


class MicroMacroModel(nn.Module):
    """
    A model for breast legion classification, inserting both the region of interest and the area around it.
    """

    def __init__(self, out_channels=2, weight=None):
        super(MicroMacroModel, self).__init__()
        self.weight = weight
        self.out_channels = out_channels
        self.features_micro = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU()

        )
        self.features_macro = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU()

        )

        self.classifier = nn.Sequential(
            nn.Linear(1168 , 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, self.out_channels)
        )

    def forward(self, micro, macro):
        micro = self.features_micro(micro)
        macro = self.features_macro(macro)


        micro = micro.view(micro.size(0), -1)
        #print('micro.shape: ', micro.shape)
        macro = macro.view(macro.size(0), -1)
        #print('macro.shape: ', macro.shape)

        x = torch.cat((micro, macro), dim=1)
        #print('x after cat shape: ', x.shape)
        x = self.classifier(x)
        return x

