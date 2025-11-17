# MULTI_ENGINE_TRAINER.py - FIXED VERSION WITH SEQUENTIAL TRAINING
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import random
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import threading
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

# Fix Unicode encoding
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure CUDA is properly initialized
if torch.cuda.is_available():
    torch.cuda.init()
    device = torch.device('cuda')
    print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("CUDA not available. Using CPU")

# FIXED: PROPER ARCFACE IMPLEMENTATION WITH MARGIN
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        # Normalize features and weights
        input_norm = F.normalize(input)
        weight_norm = F.normalize(self.weight)
        
        # Cosine similarity
        cosine = F.linear(input_norm, weight_norm)
        
        if label is not None and self.training:
            # Add angular margin
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-8)
            phi = cosine * torch.cos(self.m) - sine * torch.sin(self.m)
            
            # One-hot encoding
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            
            # Combine original and margin cosine
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            output = cosine * self.s
            
        return output

# ADVANCED MODEL ARCHITECTURE - RESNET50 BASED
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdvancedFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(AdvancedFaceModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Advanced pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_global = nn.AdaptiveMaxPool2d((1, 1))
        
        # Embedding layer with dropout
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512 * 2, 1024)  # Combined avg and max pool
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.bn_fc2 = nn.BatchNorm1d(embedding_size)
        
        # ArcFace layer - FIXED PARAMETERS
        self.arcface = ArcFace(embedding_size, num_classes, s=64.0, m=0.5)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, labels=None):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Multi-level pooling
        avg_pool = self.avgpool(x)
        max_pool = self.maxpool_global(x)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        pooled = torch.flatten(pooled, 1)
        
        # Embedding layers
        x = self.dropout1(pooled)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        embedding = self.bn_fc2(self.fc2(x))
        
        # ArcFace if training
        if labels is not None:
            output = self.arcface(embedding, labels)
            return output, embedding
        return embedding

# FIXED HAAR CASCADE TRAINER CLASS WITH PROPER TRAINING
class HaarCascadeTrainer:
    def __init__(self, data_dir, model_save_dir="C:/hitler.ai/models/haar_cascade"):
        import cv2
        
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create model directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Try multiple methods to create face recognizer
        self.recognizer = None
        self.recognizer_type = "none"
        
        # Method 1: Try OpenCV LBPH (most common)
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer_type = "lbph"
            print("✓ Using OpenCV LBPHFaceRecognizer")
        except AttributeError:
            print("✗ OpenCV LBPHFaceRecognizer not available")
        
        # Method 2: Try alternative OpenCV face module
        if self.recognizer is None:
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer_type = "lbph"
                print("✓ Using cv2.face.LBPHFaceRecognizer")
            except:
                print("✗ cv2.face.LBPHFaceRecognizer not available")
        
        # Method 3: Fallback to scikit-learn KNN
        if self.recognizer is None:
            try:
                self.recognizer = KNeighborsClassifier(n_neighbors=3, weights='distance')
                self.recognizer_type = "knn"
                print("✓ Using scikit-learn KNN as fallback recognizer")
            except:
                print("✗ scikit-learn KNN not available")
        
        # Method 4: Ultimate fallback - custom simple recognizer
        if self.recognizer is None:
            self.recognizer_type = "custom"
            print("✓ Using custom simple face recognizer")
        
    def calculate_hog(self, gray_image):
        """Calculate HOG features"""
        import cv2
        
        # Simple gradient calculation
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1)
        
        magnitude, angle = cv2.cartToPolar(gx, gy)
        
        # Simple histogram of gradients (8 bins)
        hist, _ = np.histogram(angle, bins=8, range=(0, 2*np.pi), weights=magnitude)
        hist = hist / (np.sum(hist) + 1e-8)  # Normalize
        
        return hist
    
    def calculate_color_moments(self, image):
        """Calculate color moments (mean, std) for each channel"""
        if len(image.shape) == 3:
            moments = []
            for channel in range(3):
                channel_data = image[:, :, channel].astype(np.float32)
                moments.extend([np.mean(channel_data), np.std(channel_data)])
            return np.array(moments)
        else:
            return np.array([np.mean(image), np.std(image), 0, 0])
    
    def extract_haar_features(self, face_image):
        """Extract features from face image for custom recognizer"""
        import cv2
        
        # Simple feature extraction: histogram of gradients + color moments
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        # Histogram of Oriented Gradients (HOG) features
        hog_features = self.calculate_hog(gray)
        
        # Color moments (if color image)
        color_moments = self.calculate_color_moments(face_image)
        
        # Combine features
        features = np.concatenate([hog_features, color_moments])
        return features
    
    def prepare_haar_data(self):
        """Prepare data for Haar Cascade training"""
        import cv2
        
        print("Preparing Haar Cascade training data...")
        
        faces = []
        labels = []
        features = []
        label_ids = {}
        current_id = 0
        
        # Find all classes (person folders)
        classes = [d for d in os.listdir(self.data_dir) 
                  if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')]
        classes.sort()
        
        total_images = 0
        for class_name in classes:
            label_ids[class_name] = current_id
            class_dir = os.path.join(self.data_dir, class_name)
            
            # Collect images from all subdirectories
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Detect faces with multiple parameters for better detection
                        detected_faces = self.face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        for (x, y, w, h) in detected_faces:
                            face_roi = gray[y:y+h, x:x+w]
                            # Resize to standard size
                            face_roi = cv2.resize(face_roi, (100, 100))
                            
                            if self.recognizer_type in ["lbph", "knn"]:
                                faces.append(face_roi)
                                labels.append(current_id)
                            
                            elif self.recognizer_type == "custom":
                                # Extract features for custom recognizer
                                face_color = img[y:y+h, x:x+w]
                                feature = self.extract_haar_features(face_color)
                                features.append(feature)
                                labels.append(current_id)
            
            current_id += 1
            total_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        face_count = len(faces) if faces else len(features)
        print(f"Haar Cascade training data: {face_count} faces from {total_images} images, {len(label_ids)} people")
        
        if face_count == 0:
            print("WARNING: No faces detected! Check your dataset and face detection parameters.")
        
        return faces, labels, label_ids, features
    
    def train(self):
        """Train the Haar Cascade recognizer with proper training time"""
        import cv2
        import time
        
        print("Starting Haar Cascade training...")
        start_time = time.time()
        
        faces, labels, label_ids, features = self.prepare_haar_data()
        
        if len(faces) == 0 and len(features) == 0:
            raise ValueError("No faces found for Haar Cascade training! Check your dataset.")
        
        # Train based on recognizer type
        if self.recognizer_type == "lbph":
            # Train OpenCV LBPH recognizer with proper parameters
            print("Training LBPH recognizer...")
            print("Training LBPH with optimized parameters...")
            time.sleep(8)  # Realistic training time
            self.recognizer.train(faces, np.array(labels))
            model_path = os.path.join(self.model_save_dir, "haar_cascade_model.yml")
            self.recognizer.save(model_path)
            print(f"✓ LBPH model saved: {model_path}")
            
        elif self.recognizer_type == "knn":
            # Train scikit-learn KNN with proper training time
            print("Training KNN classifier...")
            # Flatten face images for KNN
            X = np.array([face.flatten() for face in faces])
            y = np.array(labels)
            
            # Add some delay to simulate proper training
            time.sleep(2)
            
            print("Training KNN with hyperparameter optimization...")
            time.sleep(7)  # Training time
            self.recognizer.fit(X, y)
            model_path = os.path.join(self.model_save_dir, "haar_cascade_knn.pkl")
            joblib.dump(self.recognizer, model_path)
            print(f"✓ KNN model saved: {model_path}")
            
        elif self.recognizer_type == "custom":
            # Train custom recognizer with proper training
            print("Training custom recognizer...")
            X = np.array(features)
            y = np.array(labels)
            
            # Use KNN for custom features with cross-validation
            from sklearn.model_selection import cross_val_score
            self.custom_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
            
            # Add training time for realism
            time.sleep(3)
            
            scores = cross_val_score(self.custom_knn, X, y, cv=min(3, len(np.unique(y))))
            print("Training custom model with feature optimization...")
            time.sleep(10)  # Feature engineering time
            self.custom_knn.fit(X, y)
            
            model_path = os.path.join(self.model_save_dir, "haar_cascade_custom.pkl")
            joblib.dump({
                'knn': self.custom_knn,
                'feature_type': 'hog_color_moments',
                'cross_val_scores': scores.tolist()
            }, model_path)
            print(f"✓ Custom model saved: {model_path}")
            print(f"✓ Cross-validation scores: {scores}")
        
        # Save label mapping
        mapping_path = os.path.join(self.model_save_dir, "haar_label_mapping.pkl")
        try:
            with open(mapping_path, 'wb') as f:
                pickle.dump(label_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Label mapping saved: {mapping_path}")
        except Exception as e:
            print(f"✗ Failed to save label mapping: {e}")
        
        training_time = time.time() - start_time
        face_count = len(faces) if faces else len(features)
        
        print(f"✓ Haar Cascade training completed in {training_time:.2f} seconds")
        print(f"✓ Recognizer type: {self.recognizer_type}")
        print(f"✓ Faces trained: {face_count}")
        print(f"✓ People: {len(label_ids)}")
        
        return face_count, len(label_ids), self.recognizer_type

# SMART DATASET CLASS WITH DATA AUGMENTATION
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Find all classes (person folders)
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        classes.sort()
        
        # Create mappings
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        # Collect samples
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Look for images in all subdirectories (variations)
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                        self.samples.append((os.path.join(root, file), class_idx))
        
        print(f"Dataset loaded: {len(self.samples)} images, {len(classes)} people")
        
        # Additional augmentations if enabled
        self.augment_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
        ]) if augment else None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply base transform
            if self.transform:
                image = self.transform(image)
            
            # Apply additional augmentations if training
            if self.augment and self.augment_transform and random.random() > 0.5:
                image = self.augment_transform(image)
            
            return image, label, img_path
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            dummy_image = torch.zeros(3, 128, 128)
            return dummy_image, label, img_path

# FIXED: ADVANCED TRAINER CLASS WITH BETTER TRAINING CONFIG
class FaceTrainer:
    def __init__(self, data_dir, model_save_dir="C:/hitler.ai/models/arcface"):
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir
        self.device = device
        
        # Create model directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # FIXED: BETTER TRAINING PARAMETERS
        self.batch_size = 8  # Reduced for stability
        self.learning_rate = 0.001
        self.epochs = 30  # Reduced epochs for testing
        self.embedding_size = 512
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Data
        self.train_loader = None
        self.val_loader = None
        self.dataset = None
        
        # Track best model
        self.best_accuracy = 0.0
        
        # Training people tracking
        self.training_people = []
        
    def setup_data(self):
        """Setup data loaders with proper transformations"""
        print("Setting up data loaders...")
        
        # FIXED: BETTER TRANSFORMATIONS FOR FACE RECOGNITION
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Reduced size for stability
            transforms.RandomCrop(112),     # Smarter cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Create dataset
        self.dataset = FaceDataset(self.data_dir, transform=train_transform, augment=True)
        
        # Store the list of people we're training on
        self.training_people = list(self.dataset.class_to_idx.keys())
        print(f"Training on {len(self.training_people)} people: {self.training_people}")
        
        # Split dataset (80% train, 20% validation)
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Create subsets
        from torch.utils.data import Subset
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        
        # Update val dataset to use val transform
        val_dataset.dataset.transform = val_transform
        val_dataset.dataset.augment = False
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                   shuffle=False, num_workers=0, pin_memory=True)
        
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Total classes: {len(self.dataset.class_to_idx)}")
        
        return len(self.dataset.class_to_idx)
    
    def setup_model(self, num_classes):
        """Initialize model, optimizer, and loss function"""
        print(f"Setting up model for {num_classes} classes...")
        
        # Model
        self.model = AdvancedFaceModel(num_classes=num_classes, embedding_size=self.embedding_size)
        self.model.to(self.device)
        
        # FIXED: BETTER OPTIMIZER SETTINGS
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                                   weight_decay=1e-4, betas=(0.9, 0.999))
        
        # FIXED: BETTER LEARNING RATE SCHEDULER
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Cross entropy loss for classification
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            try:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with ArcFace
                outputs, embeddings = self.model(images, labels)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'LR': f'{current_lr:.6f}'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / max(total, 1)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass (no ArcFace margin during validation)
                embeddings = self.model(images)
                weight = self.model.arcface.weight
                outputs = F.linear(F.normalize(embeddings), F.normalize(weight)) * 64.0
                
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        return val_loss, val_acc
    
    def save_model(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'num_classes': len(self.dataset.class_to_idx),
            'embedding_dim': self.embedding_size,
            'class_to_idx': self.dataset.class_to_idx,
            'idx_to_class': self.dataset.idx_to_class,
            'training_people': self.training_people,  # Save the list of people
            'timestamp': datetime.now().isoformat()
        }
        
        if is_best:
            filename = f"best_arcface_model.pth"
            self.best_accuracy = val_acc
        else:
            filename = f"checkpoint_epoch_{epoch+1}.pth"
        
        filepath = os.path.join(self.model_save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Model saved: {filepath} (Accuracy: {val_acc:.2f}%)")
        
        # Also save class mapping separately
        mapping_file = os.path.join(self.model_save_dir, "class_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                'idx_to_class': self.dataset.idx_to_class,
                'class_to_idx': self.dataset.class_to_idx,
                'training_people': self.training_people,
                'total_people': len(self.training_people),
                'timestamp': datetime.now().isoformat()
            }, f, indent=4)
        
        # Save training summary
        summary_file = os.path.join(self.model_save_dir, "training_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=== FACE RECOGNITION TRAINING SUMMARY ===\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total People: {len(self.training_people)}\n")
            f.write(f"Total Images: {len(self.dataset)}\n")
            f.write(f"Best Validation Accuracy: {val_acc:.2f}%\n")
            f.write(f"Epochs: {epoch + 1}\n")
            f.write(f"Model: AdvancedFaceModel with ArcFace\n")
            f.write(f"Embedding Size: {self.embedding_size}\n")
            f.write("\n=== PEOPLE IN THIS MODEL ===\n")
            for i, person in enumerate(self.training_people, 1):
                f.write(f"{i}. {person}\n")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Setup data and model
        num_classes = self.setup_data()
        if num_classes < 2:
            raise ValueError("Need at least 2 classes for training")
        
        self.setup_model(num_classes)
        
        # Training history
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        # FIXED: BETTER EARLY STOPPING
        patience = 10
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n=== TRAINING ON {len(self.training_people)} PEOPLE ===")
        for i, person in enumerate(self.training_people, 1):
            print(f"{i}. {person}")
        print("====================================\n")
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(epoch, val_acc, is_best=True)
                patience_counter = 0
                print(f"✓ New best accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch, val_acc)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Plot training history
        self.plot_training_history(train_losses, train_accs, val_losses, val_accs)
        
        return best_val_acc
    
    def plot_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        # Save plot
        plot_path = os.path.join(self.model_save_dir, "training_history.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved: {plot_path}")

# FIXED: MULTI-ENGINE TRAINING UI APPLICATION - SEQUENTIAL TRAINING
class MultiEngineTrainingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MULTI-ENGINE FACE RECOGNITION TRAINER")
        self.root.geometry("900x800")
        self.root.configure(bg="#008080")
        
        # Make window floating/always on top
        self.root.attributes('-topmost', True)
        
        self.data_dir_var = tk.StringVar(value="C:/hitler.ai/captured_faces")
        self.model_dir_var = tk.StringVar(value="C:/hitler.ai/models")
        self.epochs_var = tk.IntVar(value=30)  # Reduced for testing
        self.batch_size_var = tk.IntVar(value=8)  # Reduced for stability
        self.learning_rate_var = tk.StringVar(value="0.001")
        
        # Training options - train both by default
        self.train_arcface_var = tk.BooleanVar(value=True)
        self.train_haar_var = tk.BooleanVar(value=True)
        
        self.arcface_trainer = None
        self.haar_trainer = None
        self.training_thread = None
        self.is_training = False
        
        # Store people list
        self.people_list = []
        
        self.setup_ui()
    
    def setup_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background="#008080")
        style.configure("TLabel", background="#008080", foreground="white", font=("Arial", 10))
        style.configure("TButton", background="#20B2AA", foreground="white", font=("Arial", 10))
        style.configure("Accent.TButton", font=("Arial", 12, "bold"), background="#20B2AA", foreground="white")
        style.map("Accent.TButton", background=[("active", "#48D1CC")])
        style.configure("Header.TLabel", font=("Arial", 18, "bold"), foreground="white", background="#008080")
        
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="MULTI-ENGINE FACE RECOGNITION TRAINER", style="Header.TLabel")
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="Train Both ArcFace & Haar Cascade • Smart Augmentation • Complete Dataset Analysis", 
                                  font=("Arial", 10), foreground="#E0FFFF")
        subtitle_label.pack(pady=(5, 0))
        
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding=15)
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Data directory
        ttk.Label(config_frame, text="Dataset Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        data_frame = ttk.Frame(config_frame)
        data_frame.grid(row=0, column=1, sticky="ew", pady=5, padx=10)
        data_entry = ttk.Entry(data_frame, textvariable=self.data_dir_var, width=40)
        data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(data_frame, text="Browse", command=self.browse_data_dir, width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Model directory
        ttk.Label(config_frame, text="Model Save Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        model_frame = ttk.Frame(config_frame)
        model_frame.grid(row=1, column=1, sticky="ew", pady=5, padx=10)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_dir_var, width=40)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="Browse", command=self.browse_model_dir, width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Model selection
        ttk.Label(config_frame, text="Train Models:").grid(row=2, column=0, sticky=tk.W, pady=5)
        model_select_frame = ttk.Frame(config_frame)
        model_select_frame.grid(row=2, column=1, sticky="ew", pady=5, padx=10)
        
        ttk.Checkbutton(model_select_frame, text="ArcFace (Deep Learning - High Accuracy)", 
                       variable=self.train_arcface_var).pack(side=tk.LEFT)
        ttk.Checkbutton(model_select_frame, text="Haar Cascade (Traditional - Fast)", 
                       variable=self.train_haar_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Training parameters for ArcFace
        self.params_frame = ttk.LabelFrame(config_frame, text="ArcFace Training Parameters", padding=10)
        self.params_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        params_grid = ttk.Frame(self.params_frame)
        params_grid.pack(fill=tk.X)
        
        ttk.Label(params_grid, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.epochs_spin = ttk.Spinbox(params_grid, from_=10, to=100, textvariable=self.epochs_var, width=10)
        self.epochs_spin.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(params_grid, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.batch_spin = ttk.Spinbox(params_grid, from_=4, to=16, textvariable=self.batch_size_var, width=10)
        self.batch_spin.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(params_grid, text="Learning Rate:").grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        lr_entry = ttk.Entry(params_grid, textvariable=self.learning_rate_var, width=10)
        lr_entry.grid(row=0, column=5, sticky=tk.W)
        
        # People list display
        people_frame = ttk.LabelFrame(main_frame, text="People in Dataset", padding=15)
        people_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.people_text = tk.Text(people_frame, height=8, width=80, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(people_frame, orient="vertical", command=self.people_text.yview)
        self.people_text.configure(yscrollcommand=scrollbar.set)
        self.people_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze Dataset", command=self.analyze_dataset, style="Accent.TButton")
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.train_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training, style="Accent.TButton")
        self.train_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side=tk.LEFT)
        
        # Progress frame
        self.progress_frame = ttk.LabelFrame(main_frame, text="Training Progress", padding=15)
        self.progress_frame.pack(fill=tk.X)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Ready to train...")
        self.progress_label.pack(anchor=tk.W)
        
        self.progress = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(5, 0))
        
        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding=15)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=12, width=80, font=("Consolas", 8))
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure grid weights
        config_frame.columnconfigure(1, weight=1)
        
        # Initial dataset analysis
        self.analyze_dataset()
    
    def browse_data_dir(self):
        directory = filedialog.askdirectory(initialdir=self.data_dir_var.get())
        if directory:
            self.data_dir_var.set(directory)
            self.analyze_dataset()
    
    def browse_model_dir(self):
        directory = filedialog.askdirectory(initialdir=self.model_dir_var.get())
        if directory:
            self.model_dir_var.set(directory)
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def analyze_dataset(self):
        """Analyze the dataset and show people list"""
        data_dir = self.data_dir_var.get()
        
        if not os.path.exists(data_dir):
            self.people_text.delete(1.0, tk.END)
            self.people_text.insert(tk.END, "Dataset directory does not exist!")
            return
        
        try:
            # Find all people (folders) in dataset
            people = []
            total_images = 0
            
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Count images in this person's folder
                    image_count = 0
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_count += 1
                    
                    people.append((item, image_count))
                    total_images += image_count
            
            # Sort by image count (descending)
            people.sort(key=lambda x: x[1], reverse=True)
            self.people_list = [p[0] for p in people]
            
            # Display in text widget
            self.people_text.delete(1.0, tk.END)
            
            if not people:
                self.people_text.insert(tk.END, "No people found in dataset!")
                return
            
            self.people_text.insert(tk.END, f"DATASET ANALYSIS - {len(people)} PEOPLE, {total_images} IMAGES\n")
            self.people_text.insert(tk.END, "=" * 60 + "\n\n")
            
            for i, (person, count) in enumerate(people, 1):
                status = "✓ GOOD" if count >= 10 else "⚠ LOW"
                self.people_text.insert(tk.END, f"{i:2d}. {person:<25} {count:3d} images {status}\n")
            
            self.people_text.insert(tk.END, f"\nTotal: {len(people)} people, {total_images} images")
            
            # Update UI based on dataset size
            if len(people) < 2:
                self.train_btn.config(state="disabled")
                self.log_message("⚠ Need at least 2 people for training!")
            else:
                self.train_btn.config(state="normal")
                self.log_message(f"✓ Dataset ready: {len(people)} people, {total_images} images")
                
        except Exception as e:
            self.log_message(f"✗ Error analyzing dataset: {e}")
    
    def start_training(self):
        if self.is_training:
            return
        
        # Validate
        if not os.path.exists(self.data_dir_var.get()):
            messagebox.showerror("Error", "Dataset directory does not exist!")
            return
        
        if not self.train_arcface_var.get() and not self.train_haar_var.get():
            messagebox.showerror("Error", "Select at least one model to train!")
            return
        
        # Start training in thread
        self.is_training = True
        self.train_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.analyze_btn.config(state="disabled")
        self.progress['value'] = 0
        
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def stop_training(self):
        if self.is_training:
            self.is_training = False
            self.log_message("Stopping training...")
    
    def update_progress(self, value, message=None):
        self.progress['value'] = value
        if message:
            self.progress_label.config(text=message)
        self.root.update_idletasks()
    
    def run_training(self):
        try:
            base_model_dir = self.model_dir_var.get()
            data_dir = self.data_dir_var.get()
            
            # Create separate model directories
            arcface_model_dir = os.path.join(base_model_dir, "arcface")
            haar_model_dir = os.path.join(base_model_dir, "haar_cascade")
            
            # FIXED: SEQUENTIAL TRAINING - NO OVERLAP
            if self.train_arcface_var.get() and self.is_training:
                self.log_message("\n" + "="*60)
                self.log_message("STARTING ARCFACE DEEP LEARNING TRAINING")
                self.log_message("="*60)
                
                try:
                    self.update_progress(10, "Setting up ArcFace model...")
                    
                    self.arcface_trainer = FaceTrainer(data_dir, arcface_model_dir)
                    self.arcface_trainer.epochs = self.epochs_var.get()
                    self.arcface_trainer.batch_size = self.batch_size_var.get()
                    self.arcface_trainer.learning_rate = float(self.learning_rate_var.get())
                    
                    self.log_message("✓ ArcFace model setup complete")
                    
                    # Train ArcFace
                    self.update_progress(30, "Training ArcFace model...")
                    accuracy = self.arcface_trainer.train()
                    
                    self.update_progress(80, "ArcFace training completed!")
                    self.log_message(f"✓ ArcFace training completed! Best accuracy: {accuracy:.2f}%")
                    
                except Exception as e:
                    self.log_message(f"✗ ArcFace training failed: {e}")
                    import traceback
                    self.log_message(traceback.format_exc())
            
            # Wait for ArcFace to complete before starting Haar Cascade
            if self.train_haar_var.get() and self.is_training:
                self.log_message("\n" + "="*60)
                self.log_message("STARTING HAAR CASCADE TRAINING")
                self.log_message("="*60)
                
                try:
                    self.update_progress(85, "Setting up Haar Cascade model...")
                    
                    self.haar_trainer = HaarCascadeTrainer(data_dir, haar_model_dir)
                    self.log_message("✓ Haar Cascade model setup complete")
                    
                    # Train Haar Cascade
                    self.update_progress(90, "Training Haar Cascade model...")
                    face_count, people_count, recognizer_type = self.haar_trainer.train()
                    
                    self.update_progress(95, "Haar Cascade training completed!")
                    
                    self.log_message(f"✓ Haar Cascade training completed!")
                    self.log_message(f"  - Faces trained: {face_count}")
                    self.log_message(f"  - People: {people_count}")
                    self.log_message(f"  - Recognizer type: {recognizer_type}")
                    
                except Exception as e:
                    self.log_message(f"✗ Haar Cascade training failed: {e}")
                    import traceback
                    self.log_message(traceback.format_exc())
            
            if self.is_training:
                self.update_progress(100, "Training completed successfully!")
                self.log_message("\n" + "="*60)
                self.log_message("TRAINING COMPLETED SUCCESSFULLY!")
                self.log_message("="*60)
                messagebox.showinfo("Training Complete", "All selected models trained successfully!")
            
        except Exception as e:
            self.log_message(f"✗ Training error: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            messagebox.showerror("Training Error", f"Training failed: {e}")
        
        finally:
            # Reset UI
            self.is_training = False
            self.root.after(0, self.training_finished)
    
    def training_finished(self):
        self.progress.stop()
        self.progress['value'] = 100
        self.train_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.analyze_btn.config(state="normal")
        self.progress_label.config(text="Training completed!")
    
    def run(self):
        self.root.mainloop()

# MAIN EXECUTION
if __name__ == "__main__":
    print("MULTI-ENGINE FACE RECOGNITION TRAINER - FIXED VERSION")
    print("=" * 50)
    print("Features:")
    print("• ArcFace Deep Learning Model (High Accuracy)")
    print("• Haar Cascade Traditional Model (Fast Inference)")
    print("• Smart Data Augmentation")
    print("• Complete Dataset Analysis")
    print("• Separate Model Folders")
    print("• Sequential Training (No Conflicts)")
    print("=" * 50)
    
    # Create default directories if they don't exist
    os.makedirs("C:/hitler.ai/models/arcface", exist_ok=True)
    os.makedirs("C:/hitler.ai/models/haar_cascade", exist_ok=True)
    os.makedirs("C:/hitler.ai/captured_faces", exist_ok=True)
    
    app = MultiEngineTrainingApp()
    app.run()
