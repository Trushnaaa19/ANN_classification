import pandas as pd
import numpy as np

df = pd.read_csv("DateFruit_Dataset.csv")
X=df.drop("Class",axis = 1)
y=df["Class"]

from sklearn.preprocessing import StandardScaler,LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test = train_test_split(X,y,random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X_train_tensor = torch.tensor(X_train_scaled,dtype =torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype =torch.float32)

y_train_tensor = torch.tensor(y_train, dtype = torch.long)
y_test_tensor = torch.tensor(y_test,dtype = torch.long)

train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = 32)

class ANN(nn.module):
    def __init__(self):
        super(ANN,self).__init__()
        
        self.model = nn.sequential(
            nn.Linear(X.shape(1),64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLu(),
            nn.Linear(64,7)
            
        )
        
def forward(self,x):
    return self.model(x)

model = ANN()
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 100
for epoch in range(epochs):
    model.train()
    
    running_loss = 0.0
    for xb,yb in train_loader:
        optimizer.zero_grad()
        
        outputs = model(xb)
        loss = criteria(outputs, yb)
        loss.backward()
        optimizer.step() # params update

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    print(f"epoch = {epoch+1}/{epochs}, loss = {train_loss}")
    
    model.eval()

total = 0
correct = 0

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb) # [0.2, 0.5, 1.3, -0.5, ..] - 7 vals
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == yb).sum().item()
        total += yb.size(0) # actual samples in each batch

print("accuracy: ", correct/total * 100)
    
    



