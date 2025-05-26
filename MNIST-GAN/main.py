import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def display_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflattened = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflattened[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
def random_noise(number_of_samples, z_dimension):
  return torch.randn(number_of_samples, z_dimension)

class Generator(nn.Module):
  def __init__(self, z_dimension, image_channel, hidden_dimension=64):
    super().__init__()
    self.z_dimension = z_dimension
    self.generator = nn.Sequential(
      nn.ConvTranspose2d(z_dimension, hidden_dimension*4, 3, 2),
      nn.BatchNorm2d(hidden_dimension*4),
      nn.ReLU(inplace=True),

      nn.ConvTranspose2d(hidden_dimension*4 , hidden_dimension*2, 4, 1),
      nn.BatchNorm2d(hidden_dimension*2),
      nn.ReLU(inplace=True),

      nn.ConvTranspose2d(hidden_dimension*2 , hidden_dimension, 3, 2),
      nn.BatchNorm2d(hidden_dimension),
      nn.ReLU(inplace=True),

      nn.ConvTranspose2d(hidden_dimension, image_channel, 4, 2),
      nn.Tanh()
    )

  def forward(self, noise):
    noise = noise.view(len(noise), self.z_dimension, 1, 1)
    return self.generator(noise)

class Discriminator(nn.Module):
    def __init__(self, image_channel, hidden_dimension=16):

        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(image_channel, hidden_dimension, 4, 2),
            nn.BatchNorm2d(hidden_dimension),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(hidden_dimension, hidden_dimension * 2, 4, 2),
            nn.BatchNorm2d(hidden_dimension*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(hidden_dimension*2, 1, 4, 2)
        )

    def forward(self, image):
        discriminator_prediction = self.discriminator(image)
        # return discriminator_prediction.view(len(discriminator_prediction), -1)
        return discriminator_prediction.view(-1)

def generator_loss(generator, discriminator, number_images, latent_dimension, device):
  noise = random_noise(number_images, latent_dimension).to(device)
  generator_image = generator(noise)
  discriminator_generator = discriminator(generator_image)
  generator_loss = Loss(discriminator_generator, torch.ones_like(discriminator_generator))
  return generator_loss

def discriminator_loss(generator, discriminator, real_images, num_images, latent_dimension, device):
    noise = random_noise(num_images, latent_dimension).to(device);
    image_generator  = generator(noise).detach()
    discriminator_generator = discriminator(image_generator )
    discriminator_real = discriminator(real_images)
    generator_loss  = Loss(discriminator_generator, torch.zeros_like(discriminator_generator))
    real_loss = Loss(discriminator_real, torch.ones_like(discriminator_real))
    return (generator_loss + real_loss) / 2
    return discriminator_loss

batch_size = 512
data = DataLoader(
                MNIST('../Data', download=True, transform=transforms.ToTensor()),
                      batch_size=batch_size,
                      shuffle=True)

Loss = nn.BCEWithLogitsLoss()
latent_dimension = 100
display_step = 500
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
device

generator = Generator(latent_dimension, 1).to(device)
generator_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
discriminator = Discriminator(1 ).to(device)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
generator = generator.apply(weights_init)
discriminator = discriminator.apply(weights_init)

num_epochs = 100
display_step = 100
iteration = 0

for epoch in range(num_epochs):
  for images, _ in data:
    num_images = len(images)
    real_images = images.to(device)

    discriminator_optim.zero_grad()
    Discriminator_loss = discriminator_loss(generator, discriminator, real_images, num_images, latent_dimension, device)
    Discriminator_loss.backward()
    discriminator_optim.step()

    generator_optim.zero_grad()
    Generator_loss = generator_loss(generator, discriminator, num_images, latent_dimension, device)
    Generator_loss.backward()
    generator_optim.step()
    if iteration % display_step ==0 :
      with torch.no_grad():
        noise =  noise = random_noise(25,latent_dimension).to(device)
        image = generator(noise)
        display_images(image)
    iteration+=1

torch.save(generator.state_dict(), 'generator.pth')

with torch.no_grad():
    noise = random_noise(25, latent_dimension).to(device)
    final_image = generator(noise)
    display_images(final_image)
