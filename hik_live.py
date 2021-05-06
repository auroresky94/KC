import cv2
import time
# hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')

cap = cv2.VideoCapture("rtsp://admin:bizarim0@192.168.0.232:554/Streaming/channels/101")

prev_time = 0
FPS = 2
while(True):

    ret, frame = cap.read()
    current_time = time.time() - prev_time

    if (ret is True) and (current_time > 1./ FPS):

        prev_time = time.time()
        resize = cv2.resize(frame, (1900,1000))
        cv2.imshow('VideoCapture', resize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) > 0:

            break



cap.release()
cv2.destroyAllWindows()
