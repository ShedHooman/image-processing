import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# CNN Model for Face Recognition
class FaceRecognitionNet(nn.Module):
    def __init__(self):
        super(FaceRecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 5)  # 5 classes for 5 people
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
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_img)
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)

# Training function
def train_model(model, train_dir, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for person_id, person_name in enumerate(os.listdir(train_dir)):
            person_dir = os.path.join(train_dir, person_name)
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                face, _ = detect_face(img)
                
                if face is not None:
                    inputs = prepare_face(face)
                    labels = torch.tensor([person_id])
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss}')

# Testing/Recognition function
def recognize_face(model, img):
    face, face_loc = detect_face(img)
    if face is not None:
        inputs = prepare_face(face)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.item(), face_loc
    return None, None

def main():
    model = FaceRecognitionNet()
    train_dir = 'dataset'
    
    # Training
    print("Starting training...")
    train_model(model, train_dir)
    
    # Save the model
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    print("Model saved!")
    
    # Testing with webcam
    cap = cv2.VideoCapture(0)
    people = os.listdir(train_dir)
    
    while True:
        ret, frame = cap.read()
        person_id, face_loc = recognize_face(model, frame)
        
        if face_loc is not None:
            x, y, w, h = face_loc
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if person_id is not None:
                cv2.putText(frame, people[person_id], (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()