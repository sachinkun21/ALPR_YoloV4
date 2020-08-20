### Apply Yolov3 for object dectection on a video

import cv2
import time
from pred_fromDNN import predict_and_draw

cap = cv2.VideoCapture('demoVid1.mp4')

number_frame = 25.0  # higher frames better quality of the video
video_size = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('alpr_detectionDemo2_ver3.avi', fourcc, number_frame, video_size)

count = 0
while True:
    ret, frame = cap.read()

    if ret:
        count+=1
        # predictions license plate

        frame = predict_and_draw(frame, crop_save=False)
        cv2.imwrite("frame{}.jpg".format(count),frame)
        # cv2.imshow("image", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()