import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained emotion detection model
model = tf.keras.models.load_model('D:/A-Compact-Embedding-for-Facial-Expression-Similarity-main/model (1).h5')  # مسیر مدل خود را وارد کنید

# Emotion labels (7 emotions corresponding to the model)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face_image = frame[y:y+h, x:x+w]

        # Resize to 48x48 for emotion prediction
        face_resized = cv2.resize(face_image, (48, 48))

        # Convert to grayscale (if required by the model)
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Normalize and reshape
        gray_face = np.expand_dims(gray_face, axis=-1)
        gray_face = np.expand_dims(gray_face, axis=0)
        gray_face = gray_face / 255.0

        # Predict emotion
        emotion_prob = model.predict(gray_face)
        emotion_index = np.argmax(emotion_prob)
        emotion = emotion_labels[emotion_index]

        # Draw bounding box and display the emotion on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame with bounding box and emotion
    cv2.imshow('Emotion Detection', frame)

    # Print emotion to console
    print("Detected Emotion: ", emotion)

    # Stop loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
