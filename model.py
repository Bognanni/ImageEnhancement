import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection that adds the input to the output 
    of the convolutional layers. If the number of input and output channels differs a convolutional 
    layer is applied to the input to match the dimensions before adding it to the output.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            # bidimensional convolutional layer with kernel size 3, padding 1 and no bias (same dimensions output)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # normalization layer with 8 groups (group normalization is more effective than batch normalization for small batch sizes)
            nn.GroupNorm(8, mid_channels),
            # activation function with negative slope of 0.2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels)
        )

        if in_channels != out_channels:
            # residual connection that matches the dimensions of the output if necessary
            self.shortcut = nn.Sequential(
                # kernel size 1 convolution to match the number of channels
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass through the residual block adding the output with the residual.
        """
        residual = self.shortcut(x)
        out = self.double_conv(x)
        return self.final_act(out + residual)


class Down(nn.Module):
    """
    Down-scaling block with a max pooling layer and a residual block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # max pooling layer with kernel size 2 and stride 2 to downsample the input by a factor of 2
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Up-scaling block that can use either bilinear upsampling or transposed convolution, followed by a residual block.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            # double the shape of the input using bilinear interpolation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            # double the shape of the input using a transposed convolution (learnable upsampling)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the up-scaling block. It upsamples the input, then pads it to match the 
        size of x2, concatenates them, and finally applies the convolutional block.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolutional block that reduces the number of channels to the desired output channels 
    using a 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CompactUNet(nn.Module):
    """
    A compact UNet architecture with residual blocks. The model consists of an encoder-decoder structure 
    with skip connections, where the encoder downsamples the input image while extracting features, 
    and the decoder upsamples the features to reconstruct the enhanced image. 
    
    The use of residual blocks helps to mitigate the vanishing gradient problem and allows for deeper 
    networks, while the final output is passed through a sigmoid activation function to ensure that 
    the pixel values are in the range [0, 1].
    """

    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(CompactUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # we use factor to reduce the number of channels in the bottleneck when using bilinear upsampling, 
        # as it does not have learnable parameters and can lead to a more efficient model
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the UNet architecture. It encodes the input image through the down-scaling path, 
        then decodes it through the up-scaling path while concatenating the corresponding features from the encoder 
        to preserve spatial information. Finally, it applies a sigmoid activation to ensure the output pixel values 
        are in the range [0, 1].
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)



class ChannelAttention(nn.Module):
    """
    Channel Attention module that computes attention weights for each channel by using both average and max pooling
    to capture different types of information. The attention weights are then applied to the input feature map.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Adaptive average pooling and max pooling to generate channel-wise descriptors wide one pixel per channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # The number of intermediate channels is reduced by a factor of 'ratio' to decrease the
        # computational cost of the attention mechanism.
        reduced_planes = max(in_planes // ratio, 1)

        # Fully connected layers to compute the attention weights for each channel. The same weights
        # are shared for both average and max pooled features.
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=reduced_planes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=reduced_planes, out_channels=in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the Channel Attention module. It computes the attention weights using both 
        average and max pooling, applies the fully connected layers and then adds the outputs together.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention module that computes attention weights for each spatial location by using both
    average and max pooling.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # The kernel size for the convolutional layer is typically set to 7 to capture a larger receptive 
        # field, but it can be set to 3 for a more compact model. The padding is set accordingly to 
        # maintain the spatial dimensions of the input
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the Spatial Attention module. It computes the attention weights using both
        average and max pooling along the channel dimension, concatenates the results, and applies a 
        convolutional layer to compute the spatial attention map.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) that combines both channel and spatial attention mechanisms 
    to enhance feature representation in the up-scaling path of the UNet architecture.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Multiply the input feature map with the channel attention weights and then with the spatial attention weights
        to enhance the feature representation before passing it through the up-scaling path of the UNet architecture.
        """
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class AttentionUp(nn.Module):
    """
    Up-scaling block that incorporates CBAM attention modules to enhance feature representation
    during the decoding process.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels, out_channels)

        # CBAM attention module applied to the skip connection features
        self.cbam = CBAM(in_channels // 2)

    def forward(self, x1, x2):
        """
        Equals to the forward pass of the standard Up block, but with an additional step where the channel 
        and spatial attention is applied to the skip connection features before concatenating them 
        with the upsampled features.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2 = self.cbam(x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
        UNet architecture with CBAM attention modules integrated into the up-scaling path to enhance feature representation 
        during the decoding process. The attention mechanism helps the model focus on important features while reconstructing 
        the output, which can lead to improved performance in low-light image enhancement tasks.

        The architecture consists of a standard UNet structure with residual blocks and an additional attention mechanism 
        in the up-scaling path. The output is passed through a sigmoid activation function to ensure that the pixel values 
        are in the range [0, 1].

        This model is designed to be more effective at enhancing low-light images by leveraging both spatial and channel-wise 
        attention mechanisms. The use of residual blocks helps to mitigate the vanishing gradient problem and allows for 
        deeper networks, while the attention modules help to focus on relevant features during reconstruction.
    """

    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = AttentionUp(1024, 512 // factor, bilinear)
        self.up2 = AttentionUp(512, 256 // factor, bilinear)
        self.up3 = AttentionUp(256, 128 // factor, bilinear)
        self.up4 = AttentionUp(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the UNet architecture equals to the standard UNet, but with the up-scaling 
        blocks replaced by AttentionUp blocks.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return torch.sigmoid(logits)


if __name__ == '__main__':
    # Test the model architectures with a dummy input to ensure they produce the expected output shape
    x = torch.randn(2, 3, 256, 256)

    baseline = CompactUNet()
    y_base = baseline(x)
    print("-> CompactUNet output shape:", y_base.shape)
    assert y_base.shape == (2, 3, 256, 256), "Error with the shape of the Baseline"

    attention_variant = AttentionUNet()
    y_att = attention_variant(x)
    print("-> AttentionUNet output shape:", y_att.shape)
    assert y_att.shape == (2, 3, 256, 256), "Error with the shape of the Attention Variant"