import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernResidualBlock(nn.Module):
    """
    Blocco Residuale ottimizzato per Image Enhancement.
    Utilizza GroupNorm per stabilità con batch size ridotti e attivazione GELU.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels)
        )

        # Shortcut per la connessione residuale
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.double_conv(x)
        return self.final_act(out + residual)


class Down(nn.Module):
    """Downscaling con MaxPool2D seguito da un ModernResidualBlock"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ModernResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling classico della UNet Baseline seguito da un blocco residuale"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ModernResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ModernResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Gestione dinamica delle dimensioni (padding se l'input non è multiplo di 2)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Convoluzione finale 1x1 per mappare le feature nei 3 canali RGB dell'immagine"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CompactUNet(nn.Module):
    """
    Modello Baseline richiesto dall'assignment.
    Un'architettura U-Net leggera con connessioni residuali moderne.
    """

    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(CompactUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = ModernResidualBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x2) if 'x2' in locals() and x.shape == x2.shape else self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)


# --- MODULI VARIANTI: ATTENTION (CBAM) ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(in_planes // ratio, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=reduced_planes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=reduced_planes, out_channels=in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class AttentionUp(nn.Module):
    """
    Modulo di Upscaling SOTA-driven: applica l'Attention Gate (CBAM)
    sulla skip connection per sopprimere il rumore strutturale prima della fusione.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ModernResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ModernResidualBlock(in_channels, out_channels)

        # Il modulo CBAM agisce sui canali della skip connection (che sono sempre in_channels // 2)
        self.cbam = CBAM(in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Allineamento spaziale
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Filtriamo le feature della skip connection prima di concatenare
        x2 = self.cbam(x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Variante Significativa richiesta dall'assignment.
    Integra i moduli CBAM nel percorso di risalita filtrando selettivamente le informazioni.
    """

    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = ModernResidualBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # Decoder potenziato con Attention Gates
        self.up1 = AttentionUp(1024, 512 // factor, bilinear)
        self.up2 = AttentionUp(512, 256 // factor, bilinear)
        self.up3 = AttentionUp(256, 128 // factor, bilinear)
        self.up4 = AttentionUp(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
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
    # Sanity check per verificare la correttezza del dimensionamento dei tensori
    print("Verifica dimensioni dei modelli...")
    x = torch.randn(2, 3, 256, 256)

    baseline = CompactUNet()
    y_base = baseline(x)
    print("-> CompactUNet output shape:", y_base.shape)
    assert y_base.shape == (2, 3, 256, 256), "Errore nelle dimensioni della Baseline"

    attention_variant = AttentionUNet()
    y_att = attention_variant(x)
    print("-> AttentionUNet output shape:", y_att.shape)
    assert y_att.shape == (2, 3, 256, 256), "Errore nelle dimensioni della Variante Attention"
    print("Tutti i controlli dimensionali sono andati a buon fine!")