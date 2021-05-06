import cv2
import time
# hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')

cap = cv2.VideoCapture("rtsp://admin:bizarim0@192.168.0.232:554/Streaming/channels/101")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('outpy.avi',fourcc, 60, (1900,1000))

while (True):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1900,1000))

    cv2.imshow('frame', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
