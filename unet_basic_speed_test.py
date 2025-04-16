import torch
import torch.nn as nn
import time  # Import the time module for measuring inference time.

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        # Encoder: First block
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),  
            nn.ReLU(inplace=True)
        )
        # Pooling reduces resolution by a factor of 2.
        self.pool = nn.MaxPool2d(2, 2)
        # Encoder: Second block; it increases the number of feature maps.
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),  
            nn.ReLU(inplace=True)
        )
        # Decoder: Up-sampling layer using transpose convolution
        self.upconv = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # Decoder: Convolutional block after upsampling.
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Output convolution: Reduces channels to the desired number of output channels.
        self.outconv = nn.Conv2d(16, out_channels, kernel_size=1)
      
    def forward(self, x):
        # Encoder forward pass:
        x1 = self.enc1(x)       # First convolution block. Output shape: (B, 16, H, W)
        x2 = self.pool(x1)      # Down-sampling: (B, 16, H/2, W/2)
        x3 = self.enc2(x2)      # Second convolution block. Output shape: (B, 32, H/2, W/2)
        # Decoder forward pass:
        x4 = self.upconv(x3)    # Upsampling: (B, 16, H, W)
        # Skip connection: We add features from the first encoder block (x1) to the upsampled output.
        x5 = self.dec1(x4 + x1)  
        out = torch.sigmoid(self.outconv(x5))  # Sigmoid squashes outputs to [0, 1].
        return out

# Testing the network and exporting to ONNX
if __name__ == "__main__":
    print('Started..')
    model = SimpleUNet()
    
    # Calculate and print the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    
    # Create a dummy input for testing and export
    x = torch.randn(1, 3, 320, 320)
    
    # Measure inference time for a single image
    start_time = time.time()
    y = model(x)
    end_time = time.time()
    
    print("Output shape:", y.shape)  # Expected output shape: (1, 1, 320, 320)
    print(f"Inference time for a single image: {end_time - start_time:.6f} seconds")  # Print inference time

    # Speed test for 10 images
    num_images = 10
    total_time = 0.0
    for i in range(num_images):
        x = torch.randn(1, 3, 320, 320)  # Generate a new random input
        start_time = time.time()
        y = model(x)
        end_time = time.time()
        total_time += (end_time - start_time)
        print(f"Inference time for image {i + 1}: {end_time - start_time:.6f} seconds")
    
    avg_time = total_time / num_images
    fps = 1 / avg_time if avg_time > 0 else 0  # Calculate FPS
    print(f"Average inference time for {num_images} images: {avg_time:.6f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")

    # Export the model to ONNX so it can be visualized in Netron
    onnx_path = "simple_unet.onnx"
    torch.onnx.export(model, x, onnx_path, 
                      input_names=["input"],
                      output_names=["output"],
                      opset_version=11,
                      do_constant_folding=True)
    print(f"Model exported to {onnx_path}. You can now visualize it in Netron (https://netron.app/).")
