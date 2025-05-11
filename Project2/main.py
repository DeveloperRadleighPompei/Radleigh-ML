import time
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########## Data ##########
data = fetch_openml(name="house_prices", as_frame=True)
df = data.frame

########## Impute ##########
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(exclude=["object"])), columns=df.select_dtypes(exclude=["object"]).columns)

########## Encode ##########
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
categorical_columns = df.select_dtypes(include=["object"]).columns
encoded_categoricals = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out(categorical_columns))

########## Prepre X and y ##########
df_processed = pd.concat([df_imputed, encoded_df], axis=1)
X = df_processed
y = df["SalePrice"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########## Scaler ##########
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1,1))

########## Convert to tensor ##########
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

########## Batching
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

########## Hyperparameters ##########
learning_rate = 0.001
input_size = X_train.shape[1]
output_size = 1
hidden_sizes = [512, 256, 128, 64, 32]
num_epochs = 1300

########## MultilayerPerceptron ##########
class MultilayerPerceptron(nn.Module):
  def __init__(self, input_size, hidden_sizes, output_size):
    super(MultilayerPerceptron, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_sizes[0]) ### Layer 1
    self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
    self.dropout1 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(hidden_sizes[0],hidden_sizes[1]) ### Layer 2
    self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
    self.dropout2 = nn.Dropout(0.3)
    self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2]) ### Layer 3
    self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
    self.dropout3 = nn.Dropout(0.2)
    self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3]) ### Layer 4
    self.bn4 = nn.BatchNorm1d(hidden_sizes[3])
    self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4]) ### Layer 5
    self.fc6 = nn.Linear(hidden_sizes[4], output_size) ### Layer 6

  def forward(self, x):
    x = F.relu(self.bn1(self.fc1(x))) ### Layer 1
    x = self.dropout1(x)
    x = F.relu(self.bn2(self.fc2(x))) ### Layer 2
    x = self.dropout2(x)
    x = F.relu(self.bn3(self.fc3(x))) ### Layer 3
    x = self.dropout3(x)
    x = F.relu(self.bn4(self.fc4(x))) ### Layer 4
    x = F.relu(self.fc5(x)) ### Layer 5
    x = self.fc6(x) ### Layer 6
    return x
  
########## Train model ##########
model = MultilayerPerceptron(input_size, hidden_sizes, output_size).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float("inf")
best_epoch = -1
start_time = time.time()

for epoch in range(num_epochs):
  model.train()
  epoch_loss = 0.0

  for batch_X, batch_y in train_loader:

    predictions = model(batch_X)
    loss = loss_fn(predictions.view(-1, 1),batch_y.view(-1, 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item() * batch_X.size(0)
  epoch_loss /= len(train_loader.dataset)
  if loss.item() < best_loss:
    best_loss = loss.item()
    best_epoch = epoch


  if (epoch + 1) % 100 == 0:
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        rmse_dollars = y_scaler.inverse_transform([[epoch_loss ** 0.5]])[0][0]

        best_rmse_dollars = y_scaler.inverse_transform([[best_loss ** 0.5]])[0][0]
        print(f"\nEpoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {epoch_loss:,.4f} | "
              f"Loss (RMSE $): ${rmse_dollars:,.2f} | "
              f"Learning Rate: {learning_rate} | "
              f"Best RMSE $: ${best_rmse_dollars:,.2f} | "
              f"Best Epoch: {best_epoch} | "
              f"Best Loss: {best_loss:,.4f} | "
              f"Time Elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")

print("##############################################################################")

torch.save(model.state_dict(), f'modelv1-{num_epochs}.pth')

########## Evaluate Model ##########
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = loss_fn(predictions, y_test_tensor)
    rmse_scaled = test_loss.item() ** 0.5
    rmse_dollars = y_scaler.inverse_transform([[rmse_scaled]])[0][0]

print(f"\nTest MSE (scaled): {test_loss.item():.4f}")
print(f"Test RMSE (scaled): {rmse_scaled:.4f}")
print(f"Test RMSE ($): ${rmse_dollars:,.2f}")

# model may show high errors due to ranged results
sample_preds = y_scaler.inverse_transform(predictions[:5].cpu().numpy())
sample_targets = y_scaler.inverse_transform(y_test_tensor[:5].cpu().numpy())
print("Sample Predictions vs Actual:")
for pred, actual in zip(sample_preds, sample_targets):
    print(f"Predicted: ${pred[0]:,.2f} | Actual: ${actual[0]:,.2f}")
