import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset_CIFAR10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset_CIFAR10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
train_dataloader = DataLoader(train_dataset_CIFAR10, batch_size=128, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset_CIFAR10, batch_size=100, shuffle=False, num_workers=2)

class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

  def forward(self,x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += self.shortcut(x)
    out = self.relu(out)
    return out

class ResNet18(nn.Module):
  def __init__(self, num_outputs=10):
    super(ResNet18, self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
    self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
    self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
    self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_outputs)


  def _make_layer(self, block, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out

model = ResNet18().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train(model, train_dataloader, optimizer, loss_function, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avgerage_loss = running_loss / total
    accuracy = correct / total
    return avgerage_loss, accuracy

def evaluate(model, test_dataloader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avgerage_loss = running_loss / total
    accuracy = correct / total
    return avgerage_loss, accuracy

num_epochs = 50

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_function, device)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} ")

torch.save(model.state_dict(), "CIFAR10-resnet18.pth")

model.load_state_dict(torch.load("CIFAR10-resnet18.pth"))
test_loss, test_accuracy = evaluate(model, test_dataloader, loss_function, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
