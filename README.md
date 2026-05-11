# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Design a neural network model to solve a regression problem using a single input feature. The model uses multiple fully connected layers with ReLU activation to predict a continuous output value. Train the network using a loss function and optimizer over several epochs to minimize error. Track and display the training loss during the learning process.

## Neural Network Model

Include the neural network model diagram.

<img width="958" height="716" alt="Screenshot 2026-04-20 143246" src="https://github.com/user-attachments/assets/8a3cb3b7-be55-437e-a71f-a243a39114e1" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: CHITLURU VENKATA SIVA DINESH KUMAR

### Register Number: 212224040055

```python
from google.colab import drive
drive.mount('/content/drive')

```

``` PY
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```PY
dataset1 = pd.read_csv('/content/drive/MyDrive/DP.csv')
X = dataset1[['input ']].values
y = dataset1[['output']].values
```

```PY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
```
```PY
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```PY
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```
```PY
# NAME : CH.V.S.DINESH KUMAR
# REG NO: 212224040055


class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```
```PY
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)
```
```PY
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      lig.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
```C
train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```C
with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
```
```C
loss_df = pd.DataFrame(ai_brain.history)
```
```C
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.show()
```
```C
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```


### Dataset Information
<img width="283" height="596" alt="image" src="https://github.com/user-attachments/assets/3dbf3e51-c2dc-4354-8c8a-8384be40e4fb" />


### OUTPUT
Training Loss Vs Iteration Plot

<img width="413" height="241" alt="image" src="https://github.com/user-attachments/assets/3031ac0b-cbd1-44dd-8a7c-25a023786823" />

<img width="853" height="582" alt="image" src="https://github.com/user-attachments/assets/f25ea5c5-1e6a-4027-8871-07f99c0e31ec" />

### New Sample Data Prediction
<img width="381" height="46" alt="image" src="https://github.com/user-attachments/assets/17f914f3-cb9a-4a8e-bbf2-3603de63e75d" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
