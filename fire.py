from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo_newdata/fire/detect/train/weights/best.pt")

print("Class Names:", model.names)

# Open video capture (replace "0" with the path to your video file)
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is None (end of video or an issue)
    if frame is None:
        print("End of video or error occurred.")
        break

    # Perform object detection on the frame
    results_list = model(frame)

    # Check if the results list is not empty
    if results_list:
        # Iterate over the results in the list
        for results in results_list:
            # Check if there are any boxes detected
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                # Convert boxes to a NumPy array
                boxes_np = results.boxes.xyxy.cpu().numpy()

                # Assuming results is a list of predictions
                for bbox in boxes_np:
                    print("bbox:", bbox)  # Print the bbox to inspect its structure
                    # Get the coordinates of the detected object
                    x, y, width, height = bbox[:4].astype(int)

                    # Decrease the dimension of the rectangular box
                    new_width = int(width * 0.8)
                    new_height = int(height * 0.8)

                    # Draw a red rectangle on the image
                    cv2.rectangle(frame, (x, y), (x + new_width, y + new_height), (0, 0, 255), 2)

                    # Label the detected object
                    if len(bbox) > 4:  # Check if bbox contains class information
                        class_index = int(bbox[5])  # Assuming class information is at index 5
                        label = model.names[class_index] if 0 <= class_index < len(model.names) else "Unknown"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the results
    cv2.imshow('YOLO Object Detection', frame)

    # Check if the 'q' key is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Increase the wait time to 100 milliseconds
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


