import torch.nn as nn
import torch.nn.functional as F


class MLAE(nn.Module):
    """
    Multi-Layer AutoEncoder module

    Parameters

    layers : list
        List of layer sizes for the encoder. The last element is the size of the latent space.
    """

    def __init__(self, layers):
        super(MLAE, self).__init__()
        self.layers = layers
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder = nn.ModuleList()
        for i in range(len(self.layers) - 1):
            encoder.append(nn.Linear(self.layers[i], self.layers[i + 1]))
        return encoder

    def build_decoder(self):
        decoder = nn.ModuleList()
        for i in range(len(self.layers) - 1, 0, -1):
            decoder.append(nn.Linear(self.layers[i], self.layers[i - 1]))
        return decoder

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
            x = F.relu(x)
        for layer in self.decoder[:-1]:
            x = F.relu(layer(x))
        return F.sigmoid(self.decoder[-1](x))

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
            x = F.relu(x)
        return x

    def decode(self, x):
        for layer in self.decoder[:-1]:
            x = F.relu(layer(x))
        return F.sigmoid(self.decoder[-1](x))


#### TODO create conditional autoencoder
