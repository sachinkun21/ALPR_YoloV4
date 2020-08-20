import cv2
import time
from ocr.easy_ocr import ret_ocr

# derive the paths to the YOLO weights and model configuration
weightsPath = "yoloV4_LPR/yolov4-LPR_train_4000.weights"
configPath = "yoloV4_LPR/yolov4-LPR_train.cfg"

net = cv2.dnn_DetectionModel(configPath, weightsPath)
net.setInputSize(416, 416)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)


with open('yoloV4_LPR/obj.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')


plate_label = 'plate'


def predict_and_draw(frame, crop_save = True):
    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    global plate_label

    if len(classes)>0:
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            # cropping image
            left, top, width, height = box
            cropped_image = frame[top:top + height, left:left + width]
            if crop_save:
                cv2.imwrite("cropped_demo{}.jpg".format(confidence), cv2.resize(cropped_image, dsize=(120,40) ))

            plate_detected = ret_ocr(cv2.resize(cropped_image, dsize=(120,40) ))

            # calling OCR function
            if plate_detected != 'plate':
                plate_label = plate_detected
            print("Detected: ", plate_label)

            label = '%.2f' % confidence
            label = '%s: %s' % (names[classId], label)

            # drawing over frame
            labelSize, baseLine = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            top = max(top, labelSize[1])
            cv2.rectangle(frame, box, color=(0, 255, 100), thickness=3)
            cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, plate_label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame

# reading Image
img = cv2.imread('demo_images/car6.jpg')

#Prediction Code
start_time = time.time()
pred_frame = predict_and_draw(img,  crop_save = True)
end_time = time.time()
print("Inference Time ",round(end_time-start_time, 2))

# saving predicted Image
print("Saving Image")
cv2.imwrite("predictions/demo6.jpg", pred_frame)



