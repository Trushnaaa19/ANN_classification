import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import torch.optim as optim

df = pd.read_csv("Powerplant_dataset.csv")
print(df.isnull().sum())

X= df.drop("PE",axis =1)
y= df["PE"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y , test_size=0.2 , random_state=42
    )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)

X_train_tensor = torch.tensor(X_train_scaled,dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype =torch.float32).view(-1,1)

X_test_tensor = torch.tensor(X_test_scaled,dtype = torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32).view(-1,1)

print(X_test_tensor)

train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size =32,shuffle = True)
test_loader = DataLoader(test_dataset, batch_size =32)
print(train_loader)

#deep learning

class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        
        self.model = nn.Sequential(
            #1st hidden layer
            nn.Linear(X_train.shape[1],6),
            nn.ReLU(),
            
            #2nd hidden layer
            nn.Linear(6,6),
            nn.ReLU(),
            
            #output layer
            nn.Linear(6,1),
            
            )
        
    def forward(self,x):
            return self.model(x)
        
model = ANN()

#loss,optimiser
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
        
#train the ANN
running_loss= 0.0
train_losses = []
val_losses = []
best_val_loss = float("inf")
epochs = 100

for epoch in range(epochs):
    model.train()
    
for xb,yb in train_loader:
    optimizer.zero_grad()
    
    outputs = model(xb)
    loss = criterion(outputs,yb)
    
    loss.backward()
    optimizer.step()
    running_loss += loss
    
epoch_train_loss = running_loss/ len(train_loader)
train_losses.append(epoch_train_loss)

#validation
model.eval()
running_val_loss = 0.0

with torch.no_grad():
    for xb,yb in test_loader:
        outputs = model(xb)
        loss = criterion(outputs,yb)
        running_val_loss += loss.item()
        
epoch_val_loss = running_val_loss/len(test_loader)
val_losses.append(epoch_val_loss)

print(f"Epoch {epoch+1}/{epochs} ==> train loss = {epoch_train_loss} & val loss = {epoch_val_loss}")

if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_model.pt") #.pt or .pth
        
# Loading the best model
model.load_state_dict(torch.load("best_model.pt"))

# Evaluation

model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    train_mse_loss = criterion(train_preds, y_train_tensor)
    test_mse_loss = criterion(test_preds, y_test_tensor)

print("Training MSE:", train_mse_loss.item())
print("Testing MSE:", test_mse_loss.item())

from sklearn.metrics import r2_score

print("r^2 score =", r2_score(y_test, test_preds))

predicted_df = pd.DataFrame(test_preds.numpy(), columns=["Predicted Values"])
actual_df = pd.DataFrame(y_test.values, columns=["Actual Values"])

pd.concat([predicted_df, actual_df], axis=1)