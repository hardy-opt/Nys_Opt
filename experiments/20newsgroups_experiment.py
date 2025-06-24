"""
MNIST classification experiment using Nys-Newton optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import yaml
from pathlib import Path

from nys_newton import NysNewtonOptimizer
from nys_newton.utils import ExperimentLogger, compute_gradient_norm

class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MNISTExperiment:
    """MNIST experiment runner."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = ExperimentLogger(self.config['logging']['log_dir'])
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        """Setup MNIST data loaders."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['training']['test_batch_size'], 
            shuffle=False
        )
    
    def setup_model(self):
        """Setup model and optimizer."""
        self.model = MNISTNet().to(self.device)
        
        optimizer_config = self.config['optimizer']
        if optimizer_config['name'] == 'NysNewton':
            self.optimizer = NysNewtonOptimizer(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                rank=optimizer_config.get('rank', 50),
                damping=optimizer_config.get('damping', 1e-6)
            )
        elif optimizer_config['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            
            # Log gradient norm
            grad_norm = compute_gradient_norm(self.model.parameters())
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log metrics
            step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.log_metric('train_loss', loss.item(), step)
                self.logger.log_metric('grad_norm', grad_norm, step)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        return test_loss, accuracy
    
    def run(self):
        """Run the complete experiment."""
        best_accuracy = 0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Test
            test_loss, test_acc = self.test()
            
            # Log epoch metrics
            self.logger.log_metric('epoch_train_loss', train_loss, epoch)
            self.logger.log_metric('epoch_train_accuracy', train_acc, epoch)
            self.logger.log_metric('epoch_test_loss', test_loss, epoch)
            self.logger.log_metric('epoch_test_accuracy', test_acc, epoch)
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), 
                          f"{self.config['logging']['log_dir']}/best_model.pt")
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save final metrics
        self.logger.save_metrics(f"{self.config['logging']['log_dir']}/metrics.pt")

def main():
    parser = argparse.ArgumentParser(description='MNIST Nys-Newton Experiment')
    parser.add_argument('--config', type=str, default='configs/mnist_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    experiment = MNISTExperiment(args.config)
    experiment.run()

if __name__ == '__main__':
    main()