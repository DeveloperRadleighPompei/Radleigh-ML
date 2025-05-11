from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time
# use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## Data ########## 
data = fetch_california_housing()
df = pd.DataFrame(data.data,columns=data.feature_names)
df["Target"] = data.target 
X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########## Scaling ##########
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

########## Convert data to correct formats ##########
def to_tensor(data):
  return torch.tensor(data, dtype=torch.float32).to(device)
X_train = to_tensor(X_train)
X_test = to_tensor(X_test)
y_train = to_tensor(y_train.to_numpy()).unsqueeze(1)
y_test = to_tensor(y_test.to_numpy()).unsqueeze(1)

########## Hyperparameters ##########
input_size = X.shape[1]
hidden_sizes = [128, 64, 32, 16]
output_size = 1
num_epochs = 6000
learning_rate = 0.001

########## Create Neural network ##########
class MultilayerPerceptron(nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
    super(MultilayerPerceptron, self).__init__()
    self.fc1 = nn.Linear(input_size,hidden_sizes[0])
    self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
    self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
    self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
    self.fc5 = nn.Linear(hidden_sizes[3], output_size)

    self.dropouts = nn.ModuleList([nn.Dropout(p=0.2) for _ in range(4)])
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropouts[0](x)
    x = F.relu(self.fc2(x))
    x = self.dropouts[1](x)
    x = F.relu(self.fc3(x))
    x = self.dropouts[2](x)
    x = F.relu(self.fc4(x))
    x = self.dropouts[3](x)
    x = self.fc5(x)
    return x

model = MultilayerPerceptron(input_size, hidden_sizes, output_size)
model.to(device)

loss_fn = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

########## Training ##########
start_time = time.time()
for epoch in range(num_epochs):
  model.train()
  predictions = model(X_train) 
  loss = loss_fn(predictions, y_train)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  elapsed_time = time.time() - start_time
  if (epoch+1) % 100 == 0:
    epoch_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    epoch_hours, epoch_rem = divmod(epoch_time, 3600)
    epoch_minutes, epoch_seconds = divmod(epoch_rem, 60)
    train_RMSE_in_100k = torch.sqrt(loss).item()
    train_RMSE_in_usd = train_RMSE_in_100k * 100000
    print(f"\nEpoch [{epoch+1}/{num_epochs}] | "
          f"Loss: {loss.item():.4f} | "
          f"Accuracy: ${train_RMSE_in_usd:,.2f} USD |"
          f"Learning Rate: {learning_rate} | "
          f"Time Elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s | "
          f"Epoch Time: {int(epoch_hours)}h {int(epoch_minutes)}m {int(epoch_seconds)}s"
          )
print("##############################################################################")
########## Save Model ##########
torch.save(model, f'modelv18-{num_epochs}')

########## Evaluate model ##########
model.eval()
with torch.no_grad():
  predictions = model(X_test)
  test_loss = loss_fn(predictions, y_test)

RMSE_in_100k = torch.sqrt(test_loss).item()
RMSE_in_usd = RMSE_in_100k * 100000
print(f"Test MSE: {test_loss.item():.4f}")
print(f"Test RMSE: {RMSE_in_100k:.4f} (in 100,000 USD)")
print(f"Test RMSE: ${RMSE_in_usd:,.2f} USD")
