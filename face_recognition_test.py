import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# CNN Model for Face Recognition
class FaceRecognitionNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]
        return face, (x, y, w, h)
    return None, None

def prepare_face(face_img):
    if face_img.shape[0] == 0 or face_img.shape[1] == 0:
        return None
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_img)
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)

def load_dataset(dataset_dir):
    images = []
    labels = []
    label_names = []
    
    for idx, person_name in enumerate(sorted(os.listdir(dataset_dir))):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            label_names.append(person_name)
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        face, _ = detect_face(img)
                        if face is not None:
                            face_tensor = prepare_face(face)
                            if face_tensor is not None:
                                images.append(face_tensor)
                                labels.append(idx)
    
    return images, labels, label_names

def train_model(model, dataset_dir, device, num_epochs=50):
    images, labels, label_names = load_dataset(dataset_dir)
    
    if not images:
        print("No valid face images found in the dataset!")
        return None
    
    # Convert lists to tensors
    images = torch.cat(images, dim=0).to(device)
    labels = torch.tensor(labels).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).float().mean()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    return label_names

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset directory
    dataset_dir = 'dataset'
    
    # Count number of classes (persons)
    num_classes = len([name for name in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, name))])
    print(f"Number of persons detected: {num_classes}")
    
    # Initialize model
    model = FaceRecognitionNet(num_classes)
    
    # Training
    print("Starting training...")
    label_names = train_model(model, dataset_dir, device)
    
    if label_names is None:
        print("Training failed! Please check your dataset.")
        return
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_names': label_names
    }, 'face_recognition_model.pth')
    print("Model saved!")
    
    # Switch to evaluation mode
    model.eval()
    
    # Testing with webcam
    print("Starting webcam for testing...")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        face, face_loc = detect_face(frame)
        
        if face_loc is not None and face is not None:
            x, y, w, h = face_loc
            face_tensor = prepare_face(face)
            
            if face_tensor is not None:
                face_tensor = face_tensor.to(device)
                with torch.no_grad():
                    outputs = model(face_tensor)
                    _, predicted = torch.max(outputs, 1)
                    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
                    
                    if confidence > 0.7:  # Confidence threshold
                        person_name = label_names[predicted.item()]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{person_name} ({confidence:.2f})", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()