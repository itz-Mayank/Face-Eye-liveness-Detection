import cv2
import os

# Create directories to save images
os.makedirs("dataset/train/faces", exist_ok=True)
os.makedirs("dataset/train/eyes", exist_ok=True)
os.makedirs("dataset/test/faces", exist_ok=True)
os.makedirs("dataset/test/eyes", exist_ok=True)

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

face_count, eye_count = 0, 0  # Counters for images saved
collecting = True

print("Press 's' to save data and 'q' to quit.")

while collecting:
    ret, img = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Save face region
        face_img = gray[y:y + h, x:x + w]
        face_path = f"dataset/train/faces/face_{face_count}.jpg"
        cv2.imwrite(face_path, face_img)
        face_count += 1

        # Detect eyes within the face region
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_path = f"dataset/train/eyes/eye_{eye_count}.jpg"
            cv2.imwrite(eye_path, eye_img)
            eye_count += 1

        # Draw rectangles around faces and eyes
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Display the frame
    cv2.imshow("Data Collection", img)

    # Handle key events
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):  # Quit
        collecting = False
    elif key == ord('s'):  # Save
        print(f"Collected {face_count} faces and {eye_count} eyes.")

cap.release()
cv2.destroyAllWindows()
