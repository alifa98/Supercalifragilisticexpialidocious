import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model.generator import Generator

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.0002
batch_size = 10
image_size = 784 # 28x28
hidden_size = 256
latent_size = 64

def generate_digit(G):
    G.load_state_dict(torch.load('G.pth'))
    G.eval()

    z = torch.randn(batch_size, latent_size).to(device)
    images = G(z)

    # save images
    torchvision.utils.save_image(images.view(images.size(0), 1, 28, 28), 'out/sample.png')

    # display images
    images = images.cpu().detach().numpy().reshape(-1, 28, 28)
    for i in range(10):  # 10 sample digits
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i], cmap='gray')
    plt.show()

# Generate digit
G = Generator(image_size, hidden_size=hidden_size, latent_size=latent_size).to(device)
generate_digit(G)
