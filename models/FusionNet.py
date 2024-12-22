import torch
import torch.nn as nn
from models.CNN import CNN5
from models.FusionCNN import FusionCNN5


class FusionNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.b10_CNN = FusionCNN5(in_channels, num_classes)
        self.b11_CNN = FusionCNN5(in_channels, num_classes)
        self.b7_CNN = FusionCNN5(in_channels, num_classes)
        self.b6_CNN = FusionCNN5(in_channels, num_classes)
        self.b76_CNN = FusionCNN5(in_channels, num_classes)

        # Fully connected layers for fusion
        # cnn_out_features = self.b10_CNN.classifier[0].in_features  # Access first Linear layer's input features
        cnn_out_features = 2
        # Final fusion CNN (includes avgpool and classifier)
        self.fusion_CNN = CNN5(5 * cnn_out_features, num_classes=num_classes, final=True)
        """
        self.fc = nn.Sequential(
            nn.Linear(5 * cnn_out_features, 512),  # Combine features from all backbones
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )"""

    def forward(self, inputs):
        # Process each input branch through its respective CNN
        b10_out = self.b10_CNN(inputs['b10'])
        b11_out = self.b11_CNN(inputs['b11'])
        b7_out = self.b7_CNN(inputs['b7'])
        b6_out = self.b6_CNN(inputs['b6'])
        b76_out = self.b76_CNN(inputs['b76'])

        # Concatenate the outputs
        fused_features = torch.cat((b10_out, b11_out, b7_out, b6_out, b76_out), dim=1)

        # Final classification
        # output = self.fc(fused_features)
        output = self.fusion_CNN(fused_features)
        
        return output
