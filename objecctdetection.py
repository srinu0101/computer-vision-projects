import cv2  # Import the OpenCV library for computer vision tasks
import time  # Import the time module to add delays
import imutils  # Import the imutils library for easier image processing

# Initialize the camera. Change the ID to 0 if 1 doesn't work.
cam = cv2.VideoCapture(0)  
if not cam.isOpened():
    print("Error: Camera not accessible")
    exit()

# Give the camera some time to warm up.
time.sleep(1)

# Variable to store the first frame for comparison.
firstframe = None
# Minimum area size for detected movements to be considered.
area = 500

while True:
    # Read the current frame from the camera.
    ret, img = cam.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break

    text = "Normal"  # Default text when no movement is detected.
    
    # Resize the image to 800 pixels wide for easier processing.
    img = imutils.resize(img, width=800)
    
    # Convert the image to grayscale for simpler processing.
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and detail.
    gaussianimg = cv2.GaussianBlur(grayimage, (21, 21), 0)

    # Set the first frame if it's not already set.
    if firstframe is None:
        firstframe = gaussianimg
        continue

    # Calculate the absolute difference between the first frame and the current frame.
    imgdiff = cv2.absdiff(firstframe, gaussianimg)
    
    # Apply a threshold to get a binary image (black and white).
    threshimg = cv2.threshold(imgdiff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate the image to fill in small holes.
    threshimg = cv2.dilate(threshimg, None, iterations=2)

    # Find contours (outlines) of the objects in the threshold image.
    cnts = cv2.findContours(threshimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Loop through each contour found.
    for c in cnts:
        # Ignore small contours that are likely noise.
        if cv2.contourArea(c) < area:
            continue
        
        # Get the bounding box for the contour.
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Draw a rectangle around the detected object.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Update the status text to indicate a moving object is detected.
        text = "Moving object detected"

    # Display the status text on the image.
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the processed video feed.
    cv2.imshow("Camera Feed", img)

    # Break the loop if the 'q' key is pressed.
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows.
cam.release()
cv2.destroyAllWindows()
