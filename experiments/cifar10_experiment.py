"""
CIFAR-10 classification experiment using Nys-Newton optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import yaml
from pathlib import Path

from nys_newton import NysNewtonOptimizer
from nys_newton.utils import ExperimentLogger, compute_gradient_norm

class CIFAR10Experiment:
    """CIFAR-10 experiment runner."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = ExperimentLogger(self.config['logging']['log_dir'])
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        """Setup CIFAR-10 data loaders."""
        # Data transformations
        if self.config['data']['augmentation']:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Datasets
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transform)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['training']['test_batch_size'], 
            shuffle=False,
            num_workers=4
        )
    
    def setup_model(self):
        """Setup model and optimizer."""
        # Model selection
        if self.config['model']['architecture'] == 'ResNet18':
            self.model = models.resnet18(pretrained=False, num_classes=10)
        elif self.config['model']['architecture'] == 'ResNet34':
            self.model = models.resnet34(pretrained=False, num_classes=10)
        else:
            raise ValueError(f"Unknown architecture: {self.config['model']['architecture']}")
        
        self.model = self.model.to(self.device)
        
        # Optimizer setup
        optimizer_config = self.config['optimizer']
        if optimizer_config['name'] == 'NysNewton':
            self.optimizer = NysNewtonOptimizer(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                rank=optimizer_config.get('rank', 100),
                damping=optimizer_config.get('damping', 1e-6)
            )
        elif optimizer_config['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['name'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 5e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['name']}")
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.1
        )
    
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
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Compute gradient norm
            grad_norm = compute_gradient_norm(self.model.parameters())
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Logging
            step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.log_metric('train_loss', loss.item(), step)
                self.logger.log_metric('grad_norm', grad_norm, step)
                
                progress = 100. * batch_idx / len(self.train_loader)
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.train_loader)} '
                      f'({progress:.0f}%)]\tLoss: {loss.item():.6f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.cross_entropy(output, target, reduction='sum')
                test_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
    
    def run(self):
        """Run the complete experiment."""
        best_accuracy = 0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Test
            test_loss, test_acc = self.test()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch metrics
            self.logger.log_metric('epoch_train_loss', train_loss, epoch)
            self.logger.log_metric('epoch_train_accuracy', train_acc, epoch)
            self.logger.log_metric('epoch_test_loss', test_loss, epoch)
            self.logger.log_metric('epoch_test_accuracy', test_acc, epoch)
            self.logger.log_metric('learning_rate', self.scheduler.get_last_lr()[0], epoch)
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                if self.config['model']['save_model']:
                    torch.save(self.model.state_dict(), 
                              self.config['model']['save_path'])
            
            print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
                  f'Best: {best_accuracy:.2f}%')
        
        # Save final metrics
        self.logger.save_metrics(f"{self.config['logging']['log_dir']}/metrics.pt")
        print(f'Best Test Accuracy: {best_accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Nys-Newton Experiment')
    parser.add_argument('--config', type=str, default='configs/cifar10_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    experiment = CIFAR10Experiment(args.config)
    experiment.run()

if __name__ == '__main__':
    main()