import cv2
import time
# hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')

cap = cv2.VideoCapture("rtsp://admin:bizarim0@192.168.123.16:554/Streaming/channels/101")


while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('VideoCapture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()