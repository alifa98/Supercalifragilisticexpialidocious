import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_size, hidden_size=256, latent_size=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, image_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
