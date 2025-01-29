import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

class RealtimeEmotionDetector:
    def __init__(self, model_path):
        """
        Initialize the real-time emotion detector
        
        Args:
            model_path (str): Path to the .h5 model file
        """
        self.model = load_model(model_path)
        self.emotions = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt']
        
        # Load face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def preprocess_frame(self, frame, face_coords):
        """
        Preprocess the detected face for model inference
        
        Args:
            frame (numpy.ndarray): Input frame
            face_coords (tuple): Coordinates of detected face (x, y, w, h)
            
        Returns:
            numpy.ndarray: Preprocessed face array
        """
        x, y, w, h = face_coords
        face = frame[y:y+h, x:x+w]
        
        # Resize to model input size
        face = cv2.resize(face, (96, 96))
        face = img_to_array(face)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def run(self):
        """
        Start real-time emotion detection using webcam
        """
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Preprocess face
                face = self.preprocess_frame(frame, (x, y, w, h))
                
                # Get prediction
                predictions = self.model.predict(face, verbose=0)
                emotion_index = np.argmax(predictions[0])
                confidence = predictions[0][emotion_index]
                
                # Get emotion label
                emotion = self.emotions[emotion_index]
                
                # Display emotion and confidence
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Real-time Emotion Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize detector with your trained model
    detector = RealtimeEmotionDetector("emotiondetector(1.8GB).h5")
    
    # Start real-time detection
    detector.run()

if __name__ == "__main__":
    main()