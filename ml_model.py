import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=27, hidden_size=64):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = BinaryClassifier()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example of training step (assuming you have data)
def train_step(X_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

#Load training data for stable_zerofluxfac
training_data = np.load('train_data_stable_zerofluxfac.npz')
X_zerofluxfac = training_data['X_zerofluxfac']
unstable_zerofluxfac = training_data['unstable_zerofluxfac']

#Load training data for stable_oneflavor
training_data2 = np.load('train_data_stable_oneflavor.npz')
X_oneflavor = training_data2['X_oneflavor']
unstable_oneflavor = training_data2['unstable_oneflavor']

#Load training data for random
training_data3 = np.load('train_data_random.npz')
X_random = training_data3['X_random']
unstable_random = training_data3['unstable_random']

#Combine the datasets
input_data = np.concatenate((X_zerofluxfac, X_oneflavor, X_random), axis=0)
output_data = np.concatenate((unstable_zerofluxfac, unstable_oneflavor, unstable_random), axis=0)

print("X_zerofluxfac.shape:", X_zerofluxfac.shape)
print("unstable_zerofluxfac.shape:", unstable_zerofluxfac.shape)
print("X_oneflavor.shape:", X_oneflavor.shape)
print("unstable_oneflavor.shape:", unstable_oneflavor.shape)
print("input_data.shape:", input_data.shape)
print("output_data.shape:", output_data.shape)

# Example usage
X_sample = torch.from_numpy(input_data) #torch.Tensor(X_zerofluxfac)  #  (sims, X)
y_sample = torch.from_numpy(output_data)  # (sims, unstable parameter value)

N_epochs = 1000
print_every_epoch = 100
for i in range(N_epochs):
    loss = train_step(X_sample, y_sample)
    if i % print_every_epoch == 0:
        print("Epoch {}, Loss = {}".format(i, loss))

