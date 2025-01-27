import cv2  # opencv-python

# Load the Haar Cascade for face detection
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Verify if the Haar Cascade is loaded correctly
if haar_cascade.empty():
    print("Error: Could not load Haar Cascade file.")
    exit()

# Initialize the camera
cam = cv2.VideoCapture(0)

# Set high resolution for better detection
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Infinite loop for real-time face detection
while True:
    ret, img = cam.read()  # Read frame from the camera
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale
    
    # Adjust parameters for better multi-face detection
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # Debugging: Check how many faces are detected
    print(f"Faces detected: {len(faces)}")

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Rectangle thickness is 3

    # Display the frame
    cv2.imshow("Face Detection", img)

    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(10)
    if key == 27:  # ASCII for 'Esc'
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
