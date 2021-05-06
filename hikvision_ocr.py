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


def draw_boxes(image, bounds, width=2):
    # image= cv2.imread(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        # print(p0)
        cv2.rectangle(image, (int(p0[0]),int(p0[1])) , (int(p2[0]),int(p2[1])), (0,255,0), width)
        # image.save(path+'\\'+'draw_'+file)
        # im2=cv2.imread(path+'\\'+'draw_'+file)
        input_text= bound[1]
        coordinates=(int(round(p3[0],0)),int(round(p3[1],0)))
        image=cv2.putText(image,input_text, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    # cv2.imwrite(path+'\\'+'draw_'+file,image)
    return image









hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')

cap = cv2.VideoCapture("rtsp://admin:bizarim0@192.168.0.232:554/Streaming/channels/101")





while(True):

    ret, frame = cap.read()


    resize = cv2.resize(frame, (1900,1000))
    # print(file)
    # image = os.path.join(path,file)

    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12,12))
    # img2=clahe.apply(img)

    # im = PIL.Image.open(image)
    reader = bizarimocr.Reader(['en'],gpu = True)
    bounds = reader.readtext(resize,allowlist=['1','2','3','4','5','6','7','8','9','0'])
    A=[]
    for i in range(len(bounds)):
        A.append(bounds[i][1])
    B = " ".join(A)
    # print(B)
    # f= open(path+"\\"+file.split(".")[0]+'.txt','w')
    # f.write(B)

    image=draw_boxes(resize, bounds)
    # p3=(int(round(p3[0],0)),int(round(p3[1],0)))
    # print(B)
    # print(p3)

    # cv2.imwrite(path+'\\'+'draw_'+file,im2)

    # decoded=decode(image)
    #
    # for d in decoded:
    #     # print(d)
    #     x, y, w, h = d.rect
    #
    #     barcode_data = d.data.decode("utf-8")
    #     barcode_type = d.type
    #
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    #     text = '%s (%s)' % (barcode_data, barcode_type)
    #     cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)



    cv2.imshow('VIDEO', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
