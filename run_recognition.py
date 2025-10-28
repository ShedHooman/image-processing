import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Model definition - harus sama dengan yang digunakan saat training
class ImprovedFaceRecognitionNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedFaceRecognitionNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Transform untuk preprocessing gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Add margin
        margin = int(0.1 * w)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        face = img[y:y+h, x:x+w]
        return face, (x, y, w, h)
    return None, None

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load saved model
    checkpoint = torch.load('face_recognition_model_improved.pth', map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Initialize model
    model = ImprovedFaceRecognitionNet(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # For prediction smoothing
    pred_history = []
    history_size = 5
    
    print("Starting face recognition... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert BGR to RGB for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face, face_loc = detect_face(rgb_frame)
        
        if face is not None and face_loc is not None:
            x, y, w, h = face_loc
            
            # Prepare face for model
            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil)
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
        
        # Display frame
        cv2.imshow('Face Recognition', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()