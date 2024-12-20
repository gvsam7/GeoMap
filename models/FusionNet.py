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
        b10_features, _ = self.b10_CNN(inputs['b10'])
        b11_features, _ = self.b11_CNN(inputs['b11'])
        b7_features, _ = self.b7_CNN(inputs['b7'])
        b6_features, _ = self.b6_CNN(inputs['b6'])
        b76_features, _ = self.b76_CNN(inputs['b76'])

        # Print the shape of the features for each branch
        print(f"b10_features shape: {b10_features.shape}")
        print(f"b11_features shape: {b11_features.shape}")
        print(f"b7_features shape: {b7_features.shape}")
        print(f"b6_features shape: {b6_features.shape}")
        print(f"b76_features shape: {b76_features.shape}")

        # Concatenate the extracted features
        fused_features = torch.cat((b10_features, b11_features, b7_features, b6_features, b76_features), dim=1)
        # Print the shape of the fused features
        print(f"Fused features shape: {fused_features.shape}")

        # Final classification
        output = self.fc(fused_features)
        # Print the final output shape
        print(f"Output shape: {output.shape}")

        return output
