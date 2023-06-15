import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model.generator import Generator
from model.discriminator import Discriminator

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.0002
batch_size = 64
image_size = 784 # 28x28
hidden_size = 256
latent_size = 64
num_epochs = 200

# Load Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)

# MNIST dataset
mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
# Data loader
dataloader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)


# Initialize networks
G = Generator(image_size, hidden_size=hidden_size, latent_size=latent_size).to(device)
D = Discriminator(image_size=image_size, hidden_size=hidden_size).to(device)

# Setup Optimizers
optim_G = optim.Adam(G.parameters(), lr=learning_rate)
optim_D = optim.Adam(D.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# For storing the best model
best_D_loss = float('inf')
best_G_loss = float('inf')

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1).to(device)
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ====== Train Discriminator ====== #
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Generate fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)

        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        optim_D.step()

        # ====== Train Generator ====== #
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        G.zero_grad()
        g_loss.backward()
        optim_G.step()

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))

    # Checkpointing the best model
    if d_loss.item() < best_D_loss or g_loss.item() < best_G_loss:
        torch.save(G.state_dict(), 'G.pth')
        torch.save(D.state_dict(), 'D.pth')
        best_D_loss = d_loss.item()
        best_G_loss = g_loss.item()
