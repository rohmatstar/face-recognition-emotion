import cv2
from deepface import DeepFace

# 1. Load the Haar cascade classifier XML file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

while True:
    # 3. Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 4. Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 5. Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # 6. Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # 7. Use DeepFace to analyze the face ROI for emotion detection
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result['dominant_emotion']
            
            # 8. Draw a rectangle around the detected face and label it with the predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
        except Exception as e:
            print(f"Error processing face ROI: {e}")
    
    # 9. Display the resulting frame with the labeled emotion
    cv2.imshow('Emotion Recognition', frame)
    
    # 10. Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 11. Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
