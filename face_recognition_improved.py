import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Custom Dataset
class FaceDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        self.load_dataset()
        print(f"Total images loaded: {len(self.images)}")

    def load_dataset(self):
        for idx, person_name in enumerate(sorted(os.listdir(self.dataset_dir))):
            person_dir = os.path.join(self.dataset_dir, person_name)
            if os.path.isdir(person_dir):
                print(f"Loading images for {person_name}")
                self.class_names.append(person_name)
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_name)
                        try:
                            # Test if image can be opened
                            img = cv2.imread(img_path)
                            if img is not None:
                                self.images.append(img_path)
                                self.labels.append(idx)
                                print(f"Successfully loaded: {img_path}")
                            else:
                                print(f"Warning: Could not load image: {img_path}")
                        except Exception as e:
                            print(f"Error loading image {img_path}: {str(e)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face, _ = detect_face(image)
            if face is None:
                face = image  # Use original image if face detection fails
            
            # Convert to PIL Image and apply transformations
            if self.transform:
                face = Image.fromarray(face)
                face = self.transform(face)
            
            return face, self.labels[idx]
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            # Return a blank image in case of error
            blank_image = torch.zeros((3, 224, 224))
            return blank_image, self.labels[idx]

# Improved CNN Model using ResNet as backbone
class ImprovedFaceRecognitionNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedFaceRecognitionNet, self).__init__()
        # Use ResNet18 as backbone
        self.resnet = models.resnet18(pretrained=True)
        # Replace last fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expected input size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Face detection function
def detect_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = ((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Add margin to the face detection
        margin = int(0.1 * w)  # 10% margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        face = img[y:y+h, x:x+w]
        return face, (x, y, w, h)
    return None, None

def train_model(model, train_loader, device, num_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def real_time_recognition(model, class_names, device):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model.eval()
    
    # Moving average for prediction smoothing
    pred_history = []
    history_size = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, face_loc = detect_face(rgb_frame)
        
        if face_loc is not None and face is not None:
            x, y, w, h = face_loc
            
            # Prepare face for model
            face_pil = Image.fromarray(face)
            face_tensor = test_transform(face_pil)
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                confidence = confidence.item()
                predicted = predicted.item()
                
                # Add to prediction history
                pred_history.append(predicted)
                if len(pred_history) > history_size:
                    pred_history.pop(0)
                
                # Get most common prediction from history
                if len(pred_history) == history_size:
                    final_prediction = max(set(pred_history), key=pred_history.count)
                    confidence_threshold = 0.8
                    
                    if confidence > confidence_threshold:
                        person_name = class_names[final_prediction]
                        color = (0, 255, 0)  # Green for high confidence
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{person_name} ({confidence:.2f})", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.9, color, 2)
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset directory
    dataset_dir = 'dataset'
    
    # Create dataset
    dataset = FaceDataset(dataset_dir, transform=train_transform)
    if len(dataset) == 0:
        print("No images found in dataset!")
        return
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    num_classes = len(dataset.class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {dataset.class_names}")
    
    model = ImprovedFaceRecognitionNet(num_classes).to(device)
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': dataset.class_names
    }, 'face_recognition_model_improved.pth')
    print("Model saved!")
    
    # Start real-time recognition
    print("Starting real-time recognition...")
    real_time_recognition(model, dataset.class_names, device)

if __name__ == "__main__":
    main()