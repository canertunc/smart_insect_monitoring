import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import load_dataset
from model import get_resnet50_model

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, save_dir='models'):
    """
    Trains the ResNet-50 model, monitors performance and saves the best model.
    
    Args:
        model (nn.Module): Model to be trained
        dataloaders (dict): Data loaders for training and validation
        criterion: Loss function
        optimizer: Optimizer
        device: Device for training (GPU/CPU)
        num_epochs (int): Number of training epochs
        save_dir (str): Directory to save the model
        
    Returns:
        model: Returns the best model
        history (dict): Contains training and validation metrics
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Dictionary to hold training and validation metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training start time
    start_time = time.time()
    
    # Save the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print("Training started...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Both training and validation phase for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iteration progress bar
            loop = tqdm(dataloaders[phase], total=len(dataloaders[phase]), leave=True)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Collect statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                loop.set_description(f'{phase.capitalize()}')
                loop.set_postfix(loss=loss.item(), acc=(torch.sum(preds == labels.data)/inputs.size(0)).item())
            
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save training and validation metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Save the best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the model with highest accuracy
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
        
        # Save the last model
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
            
        print()
    
    # Calculate total training time
    time_elapsed = time.time() - start_time
    print(f'Training completed in: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, criterion, device, class_names):
    """
    Evaluates the model and prepares classification report and confusion matrix.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader: Dataloader for evaluation
        criterion: Loss function
        device: Device for the model
        class_names (list): List of class names
        
    Returns:
        metrics (dict): Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    metrics = {
        'loss': loss,
        'accuracy': acc.item(),
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

def plot_training_history(history, save_path='results'):
    """
    Visualizes training and validation metrics.
    
    Args:
        history (dict): Dictionary of training and validation metrics
        save_path (str): Directory to save the plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    epochs = len(history['train_loss'])
    x_axis = list(range(1, epochs + 1))  # Create integer x-axis values [1, 2, 3, ...]
    
    plt.plot(x_axis, history['train_loss'], label='Training Loss')
    plt.plot(x_axis, history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(x_axis)  # Use integers for x-axis ticks
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(x_axis, history['train_acc'], label='Training Accuracy')
    plt.plot(x_axis, history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(x_axis)  # Use integers for x-axis ticks
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path='results'):
    """
    Visualizes the confusion matrix.
    
    Args:
        cm (array): Confusion matrix
        class_names (list): Class names
        save_path (str): Directory to save the plots
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def save_classification_report(report, save_path='results'):
    """
    Saves the classification report.
    
    Args:
        report (dict): Classification report
        save_path (str): Directory to save the report
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Format the report display
    metrics = ['precision', 'recall', 'f1-score', 'support']
    classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # If pandas is not available, we can save as dictionary
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write('Classification Report\n\n')
        f.write(f"{'Class':<15}")
        for metric in metrics:
            f.write(f"{metric:<15}")
        f.write('\n')
        f.write('-' * 60 + '\n')
        
        for cls in classes:
            f.write(f"{cls:<15}")
            for metric in metrics:
                f.write(f"{report[cls][metric]:<15.4f}")
            f.write('\n')
        
        f.write('-' * 60 + '\n')
        for avg in ['accuracy', 'macro avg', 'weighted avg']:
            if avg == 'accuracy':
                f.write(f"{avg:<15}")
                f.write(f"{report[avg]:<15.4f}")
                f.write('\n')
            else:
                f.write(f"{avg:<15}")
                for metric in metrics:
                    if metric != 'support' and avg == 'accuracy':
                        continue
                    f.write(f"{report[avg][metric]:<15.4f}")
                f.write('\n')

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    
    # Load dataset
    batch_size = 32
    train_loader, val_loader, dataset_info = load_dataset(data_dir='datasetv2', train_ratio=0.8, batch_size=batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print(f"Training data size: {dataset_info['train_size']}")
    print(f"Validation data size: {dataset_info['val_size']}")
    print(f"Number of classes: {dataset_info['num_classes']}")
    print(f"Class names: {dataset_info['class_names']}")
    
    # Get ResNet-50 model
    model = get_resnet50_model(num_classes=dataset_info['num_classes'], freeze_layers=True)
    model = model.to(device)
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Record training start time
    train_start_time = time.time()
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=20,  # Changed from 10 to 20
        save_dir='models'
    )
    
    # Calculate total training time
    total_train_time = time.time() - train_start_time
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders['val'],
        criterion=criterion,
        device=device,
        class_names=dataset_info['class_names']
    )
    
    # Plot training history
    plot_training_history(history, save_path='results')
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm=metrics['confusion_matrix'],
        class_names=dataset_info['class_names'],
        save_path='results'
    )
    
    # Save classification report
    save_classification_report(
        report=metrics['classification_report'],
        save_path='results'
    )
    
    print(f"Final validation loss: {metrics['loss']:.4f}")
    print(f"Final validation accuracy: {metrics['accuracy']:.4f}")
    print("All results saved to the 'results' folder.")
    print("Best model and last model saved to the 'models' folder.")

if __name__ == "__main__":
    main() 