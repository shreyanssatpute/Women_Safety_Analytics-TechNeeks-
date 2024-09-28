import cv2
import numpy as np

# Load the models
face_proto = "/Users/shreyans_satpute/Desktop/drive-download-20240828T144054Z-001/opencv_face_detector.pbtxt"
face_model = "/Users/shreyans_satpute/Desktop/drive-download-20240828T144054Z-001/opencv_face_detector_uint8.pb"
age_proto = "/Users/shreyans_satpute/Desktop/drive-download-20240828T144054Z-001/age_deploy.prototxt"
age_model = "/Users/shreyans_satpute/Desktop/drive-download-20240828T144054Z-001/age_net.caffemodel"
gender_proto = "/Users/shreyans_satpute/Desktop/drive-download-20240828T144054Z-001/gender_deploy.prototxt"
gender_model = "/Users/shreyans_satpute/Desktop/drive-download-20240828T144054Z-001/gender_net.caffemodel"

# Mean values for the models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Load the models
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# Define age and gender categories
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Histogram equalization to improve contrast
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
    frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    # Detect faces
    face_net.setInput(blob)
    detections = face_net.forward()

    face_boxes = []
    male_count = 0
    female_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.75:  # Confidence threshold
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(h/150)), 8)

    if not face_boxes:
        print("No face detected")

    for face_box in face_boxes:
        face = frame[max(0, face_box[1]-15):min(face_box[3]+15, h-1),
                     max(0, face_box[0]-15):min(face_box[2]+15, w-1)]

        # Preprocess face for gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()

        # Extract male and female probabilities
        male_prob = gender_preds[0][0]
        female_prob = gender_preds[0][1]

        # Dynamic threshold adjustment for gender classification
        # Classify as female if female probability is higher or nearly equal to male probability
        threshold_diff = 0.15  # Set to a low value to bias towards female classification
        if female_prob >= male_prob - threshold_diff:
            gender = 'Female'
            female_count += 1
        else:
            gender = 'Male'
            male_count += 1

        # Display confidence and gender
        gender_confidence = max(male_prob, female_prob)
        label = f'{gender} ({gender_confidence:.2f})'
        cv2.putText(frame, label, (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display gender counts on screen
    count_label = f'Male: {male_count}, Female: {female_count}'
    cv2.putText(frame, count_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with face boxes and gender prediction
    cv2.imshow("Age and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
