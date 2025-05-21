import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class InsectTrainer:
    def __init__(self, model, train_loader, val_loader, class_names, 
                 device='cuda', results_dir='results'):
        """
        Initialize the trainer
        
        Args:
            model (nn.Module): The model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            class_names (list): List of class names
            device (str): Device to train on ('cuda' or 'cpu')
            results_dir (str): Directory to save results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        print(f"Model {self.device} cihazına taşındı")
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Initialize best validation accuracy for model checkpointing
        self.best_val_acc = 0.0
    
    def train(self, epochs=10, lr=0.001, momentum=0.9, weight_decay=1e-4):
        """
        Train the model
        
        Args:
            epochs (int): Number of epochs to train for
            lr (float): Learning rate
            momentum (float): Momentum for SGD optimizer
            weight_decay (float): Weight decay for regularization
        
        Returns:
            history (dict): Training history
        """
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        print("Loss fonksiyonu oluşturuldu")
        
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        print("Optimizer oluşturuldu")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2
        )
        print("Learning rate scheduler oluşturuldu")
        
        print(f"Training on {self.device}")
        print(f"Total epochs: {epochs}")
        
        # Debug batch retrieval
        print(f"İlk batch alınıyor...")
        try:
            first_batch = next(iter(self.train_loader))
            print(f"İlk batch boyutu: {first_batch[0].shape}")
        except Exception as e:
            print(f"İlk batch alınırken hata: {e}")
        
        # Start training
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} başlıyor...")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()
            
            # Training loop
            batch_count = 0
            print("Eğitim döngüsü başlıyor...")
            
            for inputs, labels in self.train_loader:
                if batch_count == 0:
                    print(f"İlk batch işleniyor... Boyut: {inputs.shape}")
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                batch_count += 1
                if batch_count == 1:
                    print(f"İlk batch tamamlandı!")
                
                if batch_count % 10 == 0:
                    print(f"Batch {batch_count} işlendi")
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_acc = correct / total
            
            # Validate
            val_loss, val_acc = self._validate(criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - train_loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            
            # Save best model if validation accuracy improved
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_model(is_best=True)
                print(f"Yeni en iyi model kaydedildi! Doğrulama doğruluğu: {val_acc:.4f}")
        
        # Save final model
        self._save_model(is_best=False)
        
        # Plot and save training results
        self._plot_training_history()
        
        return self.history
    
    def _validate(self, criterion):
        """
        Validate the model
        
        Args:
            criterion: Loss function
        
        Returns:
            val_loss, val_acc
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(self.val_loader.dataset)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def evaluate(self):
        """
        Evaluate the model and generate classification report and confusion matrix
        
        Returns:
            classification_report, confusion_matrix
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate classification report
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot and save confusion matrix
        self._plot_confusion_matrix(cm)
        
        # Plot and save classification report
        self._plot_classification_report(report)
        
        # Save classification report in the requested format as a text file
        self._save_classification_report_as_text(report)
        
        return report, cm
    
    def _save_model(self, is_best=False):
        """Save the trained model"""
        if is_best:
            model_path = os.path.join(self.results_dir, 'best_vgg_insect_model.pth')
        else:
            model_path = os.path.join(self.results_dir, 'last_vgg_insect_model.pth')
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'val_acc': self.best_val_acc if is_best else self.history['val_acc'][-1]
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def _plot_training_history(self):
        """Plot and save training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_history.png'))
        print(f"Training history plot saved to {self.results_dir}/training_history.png")
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Reduce font size if there are many classes
        font_size = max(8, 12 - 0.2 * len(self.class_names))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, fontsize=font_size)
        plt.yticks(rotation=45, fontsize=font_size)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {self.results_dir}/confusion_matrix.png")
    
    def _plot_classification_report(self, report):
        """Plot and save classification report as a heatmap"""
        # Extract data from report
        report_data = []
        for class_name in self.class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1_score = report[class_name]['f1-score']
                support = report[class_name]['support']
                report_data.append([class_name, precision, recall, f1_score, support])
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(
            report_data, 
            columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        )
        
        # Create a heatmap for the metrics
        plt.figure(figsize=(10, len(self.class_names) * 0.5))
        metrics_df = df.set_index('Class')[['Precision', 'Recall', 'F1-Score']]
        sns.heatmap(
            metrics_df, annot=True, cmap='YlGnBu', fmt='.3f',
            cbar_kws={'label': 'Score'}
        )
        plt.title('Classification Report')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.results_dir, 'classification_report.png'))
        print(f"Classification report saved to {self.results_dir}/classification_report.png")
    
    def _save_classification_report_as_text(self, report):
        """Save classification report in the requested format as a text file"""
        report_path = os.path.join(self.results_dir, 'classification_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Classification Report\n\n")
            f.write("Class          precision      recall         f1-score       support        \n")
            f.write("------------------------------------------------------------\n")
            
            # Add class metrics
            for class_name in self.class_names:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1_score = report[class_name]['f1-score']
                    support = report[class_name]['support']
                    
                    # Format each line to match the required format
                    f.write(f"{class_name:<15} {precision:<15.4f} {recall:<15.4f} {f1_score:<15.4f} {support:<15.4f}\n")
            
            f.write("------------------------------------------------------------\n")
            
            # Add accuracy
            f.write(f"accuracy       {report['accuracy']:<15.4f}\n")
            
            # Add macro avg
            macro_precision = report['macro avg']['precision']
            macro_recall = report['macro avg']['recall']
            macro_f1 = report['macro avg']['f1-score']
            macro_support = report['macro avg']['support']
            f.write(f"macro avg      {macro_precision:<15.4f} {macro_recall:<15.4f} {macro_f1:<15.4f} {macro_support:<15.4f}\n")
            
            # Add weighted avg
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            weighted_support = report['weighted avg']['support']
            f.write(f"weighted avg   {weighted_precision:<15.4f} {weighted_recall:<15.4f} {weighted_f1:<15.4f} {weighted_support:<15.4f}\n")
        
        print(f"Classification report saved to {report_path}") 