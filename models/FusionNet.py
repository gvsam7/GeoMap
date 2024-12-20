import torch
import torch.nn as nn
from models.CNN import CNN5


class FusionNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.b10_CNN = CNN5(in_channels, num_classes)
        self.b11_CNN = CNN5(in_channels, num_classes)
        self.b7_CNN = CNN5(in_channels, num_classes)
        self.b6_CNN = CNN5(in_channels, num_classes)
        self.b76_CNN = CNN5(in_channels, num_classes)

        # Fully connected layers for fusion
        # cnn_out_features = self.b10_CNN.classifier[0].in_features  # Access first Linear layer's input features
        cnn_out_features = 2
        self.fc = nn.Sequential(
            nn.Linear(5 * cnn_out_features, 512),  # Combine features from all backbones
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, inputs):
        # Process each input branch through its respective CNN
        b10_out = self.b10_CNN(inputs['b10'])
        b11_out = self.b11_CNN(inputs['b11'])
        b7_out = self.b7_CNN(inputs['b7'])
        b6_out = self.b6_CNN(inputs['b6'])
        b76_out = self.b76_CNN(inputs['b76'])

        # Print the shape of the output for each branch
        print(f"b10_out shape: {b10_out.shape}")
        print(f"b11_out shape: {b11_out.shape}")
        print(f"b7_out shape: {b7_out.shape}")
        print(f"b6_out shape: {b6_out.shape}")
        print(f"b76_out shape: {b76_out.shape}")

        # Concatenate the outputs
        fused_features = torch.cat((b10_out, b11_out, b7_out, b6_out, b76_out), dim=1)
        # Print the shape of the fused features
        print(f"Fused features shape: {fused_features.shape}")

        # Final classification
        output = self.fc(fused_features)
        # Print the final output shape
        print(f"Output shape: {output.shape}")

        return output
