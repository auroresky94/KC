import cv2
import time
import os
import datetime
import threading,multiprocessing
import shutil
# hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')


# import psycopg2




def capture(cameraid,Year_Month_Date,Hour_Minute):
    if cameraid==1:
        ip='192.168.123.15'
    elif cameraid==2:
        ip='192.168.123.16'
    else:
        print('error')
        pass
    path=f'./Camera/{Year_Month_Date}'
    os.makedirs(path,exist_ok=True)
    filename=str(cameraid)+str(Hour_Minute)+'.jpg'
    cap = cv2.VideoCapture(f"rtsp://admin:bizarim0@{ip}:554/Streaming/channels/101")
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') 
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    os.makedirs('./tmp',exist_ok=True)
    ###################################################################################비디오로 저장 이미지는 에러 다수
    out = cv2.VideoWriter(f'./tmp/{cameraid}__{str(Hour_Minute)}.avi', fcc, fps, (w, h))
    n=0
    while cap.isOpened():
        n+=1
        print(n)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            if n>=100:
                cap.release()
                break
    ########################################################################################비디오에서 프레임 추출
    vidcap = cv2.VideoCapture(f'./tmp/{cameraid}__{str(Hour_Minute)}.avi')
    suc, vidframe = vidcap.read()
    count = 0
    while suc:
        if count == 12:
            cv2.imwrite(f"{path}/{filename}" , vidframe)
            break     # save frame as JPEG file      
        
        suc,vidframe = vidcap.read()
        count += 1
    vidcap.release()

    
        ################################################################판독 후 data 넣어주기
    




def main():
    while True:
        now = datetime.datetime.now()
        Year_Month_Date=now.strftime('%Y_%m_%d')
        Hour_Minute=now.strftime('%H_%M')
        try:
            t1=threading.Thread(target=capture,args=[1,Year_Month_Date,Hour_Minute])
            t2=threading.Thread(target=capture,args=[2,Year_Month_Date,Hour_Minute])
            t1.start()
            t1.join()
            t2.start()
            t2.join()
            shutil.rmtree('./tmp')
            time.sleep(100000)
        except:
            continue
    



if __name__== '__main__':
    main()