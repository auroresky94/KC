import hikvision.api
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'
from bizarimocr import *
import cv2
import PIL
from PIL import ImageDraw
import argparse
from pyzbar.pyzbar import decode
import threading

def draw_boxes(image, bounds, width=2):

    for bound in bounds:
        p0, p1, p2, p3 = bound[0]

        cv2.rectangle(image, (int(p0[0]),int(p0[1])) , (int(p2[0]),int(p2[1])), (0,255,0), width)

        input_text= bound[1]
        coordinates=(int(round(p3[0],0)),int(round(p3[1],0)))
        image=cv2.putText(image,input_text, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return image


# hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')

cap = cv2.VideoCapture("rtsp://admin:bizarim0@192.168.0.232:554/Streaming/channels/101")

while True:

    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC,
    (1000))
    resize = cv2.resize(frame, (1900,1000))

    reader = bizarimocr.Reader(['en'],gpu = True)
    bounds = reader.readtext(resize,allowlist=['1','2','3','4','5','6','7','8','9','0'])
    A=[]
    for i in range(len(bounds)):
        A.append(bounds[i][1])
    B = " ".join(A)

    image=draw_boxes(resize, bounds)
    cv2.imshow('VIDEO', image)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# threading.Timer(1, liveread(cap)).start()

cap.release()
cv2.destroyAllWindows()
