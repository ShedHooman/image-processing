import cv2
import os

def create_dataset():
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create dataset directory if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    cap = cv2.VideoCapture(0)
    
    for person in range(5):  # 5 people
        person_name = input(f"Enter name for person {person + 1}: ")
        person_dir = os.path.join('dataset', person_name)
        
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        count = 0
        print(f"Capturing images for {person_name}. Press 'c' to capture, 'q' to move to next person.")
        
        while count < 10:  # 10 images per person
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.imshow('Capture Face', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Save the face image
                if len(faces) > 0:
                    face = frame[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(person_dir, f'{count}.jpg'), face)
                    print(f"Captured image {count + 1}/10 for {person_name}")
                    count += 1
            elif key == ord('q'):
                break
        
        if count < 10:
            print(f"Warning: Only captured {count} images for {person_name}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Dataset creation completed!")

if __name__ == "__main__":
    create_dataset()