import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import matplotlib.pyplot as plt

baseDir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(baseDir, 'iris')
data_path = os.path.join(model_path, 'iris.data')

# ======================================================================
# STEP 1: CREATE THE DATASET CLASS (FIXED VERSION)
# ======================================================================
class IrisDataset(Dataset):
    """Custom Dataset for loading the Iris dataset"""
    
    def __init__(self, csv_path=None):
        super().__init__()
        
        # Load data from CSV file
        if csv_path and os.path.exists(csv_path):
            print(f"Loading data from: {csv_path}")
            
            # Read CSV without header (your file has no header row)
            # Your file has 5 columns: 4 features + 1 label
            column_names = ['sepal_length', 'sepal_width', 
                           'petal_length', 'petal_width', 'species']
            df = pd.read_csv(csv_path, header=None, names=column_names)
            
            print(f"✓ Loaded {len(df)} samples")
            print(f"✓ Data shape: {df.shape}")
            print(f"✓ First few rows:\n{df.head()}")
            
        else:
            # Fallback to sklearn dataset
            print("Using sklearn built-in dataset as fallback")
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['species'] = iris.target
            df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Store the original dataframe for reference
        self.df = df
        
        # Convert string labels to numerical values
        self.label_encoder = LabelEncoder()
        df['species_encoded'] = self.label_encoder.fit_transform(df['species'])
        
        # Extract features and labels
        features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        labels = df['species_encoded'].values
        
        print(f"✓ Unique species: {df['species'].unique()}")
        print(f"✓ Encoded labels: {np.unique(labels)}")
        
        # Normalize features (important for neural networks)
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        
        # Convert to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        print(f"✓ Features tensor shape: {self.features.shape}")
        print(f"✓ Labels tensor shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_original_sample(self, idx):
        """Get original (un-normalized) sample"""
        return self.df.iloc[idx]

# ======================================================================
# STEP 2: CREATE THE NEURAL NETWORK MODEL
# ======================================================================
class IrisClassifier(nn.Module):
    """Simple neural network for Iris classification"""
    
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisClassifier, self).__init__()
        
        # Define network layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, num_classes)
        
        print(f"✓ Created model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x):
        """Forward pass: x -> layer1 -> ReLU -> layer2 -> ReLU -> layer3"""
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def predict(self, x):
        """Make class predictions"""
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

# ======================================================================
# STEP 3: TRAINING FUNCTION
# ======================================================================
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.01):
    """Complete training loop demonstrating PyTorch fundamentals"""
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(num_epochs):
        # === TRAINING PHASE ===
        model.train()  # Set model to training mode
        epoch_train_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            # 1. Zero the gradients (important!)
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs = model(features)
            
            # 3. Compute loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass (AUTOGRAD computes gradients!)
            loss.backward()
            
            # 5. Update parameters
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # === VALIDATION PHASE ===
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation for validation
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Val Acc: {val_accuracy:6.2f}%')
    
    print("="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    return train_losses, val_losses, val_accuracies

# ======================================================================
# STEP 4: MAIN EXECUTION
# ======================================================================
def main():
    print("="*70)
    print("IRIS CLASSIFICATION - PYTORCH FUNDAMENTALS PRACTICE")
    print("="*70)
    
    # ==================================================================
    # 1. LOAD DATASET
    # ==================================================================
    print("\n1. LOADING DATASET")
    print("-" * 40)
    
    # Try loading from your iris folder
    try:
        dataset = IrisDataset(data_path)
        print("✓ Successfully loaded from local file")
    except Exception as e:
        print(f"Error loading from local file: {e}")
        print("Trying current directory...")
        dataset = IrisDataset('iris.data')
    
    # ==================================================================
    # 2. CREATE DATA LOADERS (TENSOR BATCHING)
    # ==================================================================
    print("\n2. CREATING DATA LOADERS")
    print("-" * 40)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))    # 70% for training
    val_size = int(0.15 * len(dataset))     # 15% for validation
    test_size = len(dataset) - train_size - val_size  # 15% for testing
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Training set: {len(train_dataset)} samples")
    print(f"✓ Validation set: {len(val_dataset)} samples")
    print(f"✓ Test set: {len(test_dataset)} samples")
    
    # Create DataLoaders (handles batching and shuffling)
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Show a sample batch
    sample_features, sample_labels = next(iter(train_loader))
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Feature shape per batch: {sample_features.shape}")
    print(f"✓ Label shape per batch: {sample_labels.shape}")
    
    # ==================================================================
    # 3. CREATE MODEL (NN.MODULE)
    # ==================================================================
    print("\n3. BUILDING NEURAL NETWORK")
    print("-" * 40)
    
    model = IrisClassifier(input_size=4, hidden_size=10, num_classes=3)
    print("\nModel architecture:")
    print(model)
    
    # ==================================================================
    # 4. TRAIN THE MODEL (TRAINING LOOP, AUTOGRAD, OPTIMIZER)
    # ==================================================================
    print("\n4. TRAINING THE MODEL")
    print("-" * 40)
    
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=100, learning_rate=0.01
    )
    
    # ==================================================================
    # 5. TEST THE MODEL
    # ==================================================================
    print("\n5. TESTING THE MODEL")
    print("-" * 40)
    
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for confusion matrix
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    test_accuracy = 100 * correct / total
    print(f"✓ Test Accuracy: {test_accuracy:.2f}%")
    
    # ==================================================================
    # 6. SAVE AND LOAD MODEL
    # ==================================================================
    print("\n6. SAVING AND LOADING MODEL")
    print("-" * 40)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the entire model
    torch.save(model, 'models/iris_model_complete.pth')
    print("✓ Saved complete model: models/iris_model_complete.pth")
    
    # Save only the model parameters (recommended)
    # Save the numpy arrays instead of sklearn objects
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': dataset.scaler.mean_,
        'scaler_scale': dataset.scaler.scale_,
        'label_encoder_classes': dataset.label_encoder.classes_,
    }, 'models/iris_model_state.pth')
    print("✓ Saved model state: models/iris_model_state.pth")
    
    # Demonstrate loading
    print("\nDemonstrating model loading...")
    
    # Load complete model (use weights_only=False for compatibility)
    try:
        loaded_complete_path = os.path.join(baseDir, 'models', 'iris_model_complete.pth')
        loaded_complete = torch.load(loaded_complete_path, weights_only=False)
        print("✓ Loaded complete model")
    except Exception as e:
        print(f"Note: Could not load complete model due to security restrictions: {e}")
        print("This is expected in PyTorch 2.6+ for security reasons")
    
    # Load state dict into new model (SAFE METHOD)
    checkpoint_path = os.path.join(baseDir, 'models', 'iris_model_state.pth')
    
    # Method 1: Using weights_only=False (if you trust the file)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Method 2: Alternative - use safe_globals (more secure)
    # import torch.serialization
    # with torch.serialization.safe_globals([StandardScaler, LabelEncoder]):
    #     checkpoint = torch.load(checkpoint_path)
    
    # Create new model and load state dict
    new_model = IrisClassifier()
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create new scaler and label encoder from saved parameters
    new_scaler = StandardScaler()
    new_scaler.mean_ = checkpoint['scaler_mean']
    new_scaler.scale_ = checkpoint['scaler_scale']
    new_scaler.n_features_in_ = len(checkpoint['scaler_mean'])
    new_scaler.feature_names_in_ = None
    
    new_label_encoder = LabelEncoder()
    new_label_encoder.classes_ = checkpoint['label_encoder_classes']
    
    # Test the loaded model
    new_model.eval()
    test_input = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)
    test_input_normalized = torch.tensor(
        new_scaler.transform(test_input.numpy()),
        dtype=torch.float32
    )
    with torch.no_grad():
        prediction = new_model.predict(test_input_normalized)
    
    predicted_species = new_label_encoder.inverse_transform(prediction.numpy())
    print("✓ Loaded model state into new model")
    print("✓ Recreated scaler and label encoder")
    print(f"✓ Test prediction with loaded model: {predicted_species[0]}")
    
    # ==================================================================
    # 7. GPU/CUDA DEMONSTRATION
    # ==================================================================
    print("\n7. GPU/CUDA DEMONSTRATION")
    print("-" * 40)
    
    if torch.cuda.is_available():
        print("CUDA is available! Demonstrating GPU usage...")
        device = torch.device('cuda')
        
        # Move model to GPU
        model_gpu = IrisClassifier().to(device)
        
        # Move a sample batch to GPU
        sample_features, sample_labels = next(iter(train_loader))
        sample_features_gpu = sample_features.to(device)
        sample_labels_gpu = sample_labels.to(device)
        
        # Perform forward pass on GPU
        outputs_gpu = model_gpu(sample_features_gpu)
        
        print(f"✓ Model device: {next(model_gpu.parameters()).device}")
        print(f"✓ Features device: {sample_features_gpu.device}")
        print(f"✓ Outputs device: {outputs_gpu.device}")
        
        # Move back to CPU for comparison
        outputs_cpu = outputs_gpu.cpu()
        print(f"✓ Outputs moved back to CPU: {outputs_cpu.device}")
    else:
        print("No GPU available. Running on CPU.")
    
    # ==================================================================
    # 8. VISUALIZATION
    # ==================================================================
    print("\n8. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training and validation loss
    axes[0, 0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy
    axes[0, 1].plot(val_accuracies, color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sample predictions
    axes[1, 0].bar(['Correct', 'Incorrect'], 
                  [correct, total - correct], 
                  color=['green', 'red'])
    axes[1, 0].set_title(f'Test Set Results: {test_accuracy:.1f}% Accuracy')
    axes[1, 0].set_ylabel('Number of Samples')
    
    # Plot 4: Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.label_encoder.classes_,
                yticklabels=dataset.label_encoder.classes_,
                ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('models/training_results.png', dpi=100, bbox_inches='tight')
    print("✓ Saved visualization: models/training_results.png")
    
    # ==================================================================
    # 9. MAKE PREDICTIONS ON NEW SAMPLES
    # ==================================================================
    print("\n9. MAKING PREDICTIONS")
    print("-" * 40)
    
    # Create some test samples
    test_samples = torch.tensor([
        [5.1, 3.5, 1.4, 0.2],  # Should be Iris-setosa
        [6.7, 3.0, 5.2, 2.3],  # Should be Iris-virginica
        [5.9, 3.0, 4.2, 1.5],  # Should be Iris-versicolor
    ], dtype=torch.float32)
    
    # Normalize using the same scaler
    test_samples_normalized = torch.tensor(
        dataset.scaler.transform(test_samples.numpy()),
        dtype=torch.float32
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model.predict(test_samples_normalized)
    
    # Decode predictions
    predicted_species = dataset.label_encoder.inverse_transform(predictions.numpy())
    
    print("Sample predictions:")
    for i, (sample, pred) in enumerate(zip(test_samples, predicted_species)):
        print(f"  Sample {i+1}: {sample.numpy()} → {pred}")
    
    # ==================================================================
    # 10. SUMMARY
    # ==================================================================
    print("\n" + "="*70)
    print("PYTORCH FUNDAMENTALS PRACTICED:")
    print("="*70)
    print("✓ 1. Tensors and tensor operations")
    print("✓ 2. Autograd and computational graphs (loss.backward())")
    print("✓ 3. Building models with nn.Module")
    print("✓ 4. Loss functions (CrossEntropyLoss) and optimizers (Adam)")
    print("✓ 5. Training loop from scratch")
    print("✓ 6. Using GPU (CUDA basics)")
    print("✓ 7. Save and load models")
    print("="*70)
    print("\nProject structure created:")
    print("  iris/iris.data          # Your dataset")
    print("  iris_classification.py  # This script")
    print("  models/                 # Saved models and plots")
    print("    ├── iris_model_complete.pth")
    print("    ├── iris_model_state.pth")
    print("    └── training_results.png")
    print("="*70)

if __name__ == "__main__":
    main()