# Usage Examples

This document provides comprehensive examples of using the Nys-Newton optimizer in various scenarios.

## Basic Usage

### Simple Linear Regression

```python
import torch
import torch.nn as nn
from nys_newton import NysNewtonOptimizer

# Generate synthetic data
n_samples, n_features = 1000, 20
X = torch.randn(n_samples, n_features)
true_weights = torch.randn(n_features)
y = X @ true_weights + 0.1 * torch.randn(n_samples)

# Define model
model = nn.Linear(n_features, 1, bias=False)

# Initialize optimizer
optimizer = NysNewtonOptimizer(model.parameters(), lr=0.01, rank=10)
criterion = nn.MSELoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(X).squeeze()
    loss = criterion(predictions, y)
    
    # Backward pass
    loss.backward()
    
    # Optimization step
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item():.6f}')

Neural Network Classification
pythonimport torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from nys_newton import NysNewtonOptimizer

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Setup
input_size, hidden_size, num_classes = 784, 128, 10
model = SimpleNet(input_size, hidden_size, num_classes)

# Optimizer with custom parameters
optimizer = NysNewtonOptimizer(
    model.parameters(),
    lr=0.001,
    rank=50,
    damping=1e-5,
    update_freq=5  # Update curvature every 5 steps
)

# Training function
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)
Advanced Usage
Custom Parameter Groups
python# Different learning rates for different layers
optimizer = NysNewtonOptimizer([
    {'params': model.fc1.parameters(), 'lr': 0.01},
    {'params': model.fc2.parameters(), 'lr': 0.005},
    {'params': model.fc3.parameters(), 'lr': 0.001, 'rank': 20}
], rank=50)  # Default rank for groups without specified rank
Learning Rate Scheduling
pythonfrom torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Step decay
optimizer = NysNewtonOptimizer(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Training with scheduler
for epoch in range(100):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    scheduler.step()
    
    print(f'Epoch {epoch}: Loss = {train_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')
Gradient Clipping
pythonimport torch.nn.utils as utils

max_grad_norm = 1.0

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        loss = compute_loss(model, batch)
        loss.backward()
        
        # Clip gradients
        utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
Comparison Studies
Optimizer Comparison
pythonimport matplotlib.pyplot as plt
from nys_newton import NysNewtonOptimizer

def compare_optimizers(model_class, data_loader, num_epochs=50):
    """Compare different optimizers on the same problem."""
    
    optimizers = {
        'Nys-Newton': lambda params: NysNewtonOptimizer(params, lr=0.01, rank=50),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001),
        'SGD': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
    }
    
    results = {}
    
    for name, opt_fn in optimizers.items():
        model = model_class()
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = train_epoch(model, data_loader, optimizer, criterion)
            losses.append(epoch_loss)
        
        results[name] = losses
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    return results

# Usage
results = compare_optimizers(SimpleNet, train_loader)
Hyperparameter Sensitivity
pythondef rank_sensitivity_study(model, data_loader, ranks=[10, 20, 50, 100]):
    """Study the effect of Nyström rank on convergence."""
    
    results = {}
    
    for rank in ranks:
        model_copy = model_class()
        optimizer = NysNewtonOptimizer(model_copy.parameters(), lr=0.01, rank=rank)
        
        losses = []
        for epoch in range(30):
            loss = train_epoch(model_copy, data_loader, optimizer, nn.CrossEntropyLoss())
            losses.append(loss)
        
        results[f'Rank {rank}'] = losses
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Rank Sensitivity Analysis')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()
Performance Optimization
Memory-Efficient Training
python# For large models, use smaller ranks and less frequent updates
optimizer = NysNewtonOptimizer(
    large_model.parameters(),
    lr=0.01,
    rank=20,           # Smaller rank for memory efficiency
    update_freq=20,    # Less frequent curvature updates