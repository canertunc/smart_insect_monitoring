import argparse
import torch
import os
import time
from model import create_vgg_model
from data_loader import get_data_loaders
from trainer import InsectTrainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train VGG model for insect classification')
    parser.add_argument('--data_dir', type=str, default='dataset', 
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--freeze_layers', type=int, default=5, 
                        help='Number of layers to freeze in VGG model')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to train on (cuda or cpu)')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    else:
        print(f"CUDA bilgisi: {torch.cuda.get_device_name(0)}")
        print(f"CUDA bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("Veri yükleyicileri hazırlanıyor...")
    # Get data loaders
    train_loader, val_loader, class_names, num_classes = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    print("Model oluşturuluyor...")
    # Create model
    model = create_vgg_model(
        num_classes=num_classes, 
        freeze_layers=args.freeze_layers
    )
    
    print("Eğitici hazırlanıyor...")
    # Create trainer
    trainer = InsectTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        device=args.device,
        results_dir=args.results_dir
    )
    
    print("Eğitim başlıyor...")
    # Eğitim süresini ölçmeye başla
    training_start_time = time.time()
    
    # Train model
    trainer.train(
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Eğitim süresini hesapla
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Evaluate model
    report, cm = trainer.evaluate()
    
    print("Training and evaluation complete!")
    print(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"All results saved to {args.results_dir}")

if __name__ == '__main__':
    main() 