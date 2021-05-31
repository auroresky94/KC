from sys import exec_prefix
import cv2
import time
import os
import datetime
import threading
import torch.multiprocessing as multiprocessing 
import shutil
import easyocr
# hik_camera = hikvision.api.CreateDevice('192.168.0.232', username='admin', password='bizarim0')
import numpy as np

import psycopg2

def insert_tiot_table16_total_count(org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13):
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ INSERT INTO tiot_table16 (org_cd,seq,sens_data1, sens_data2, sens_data3, sens_data6, sens_data7, sens_data8, sens_data9, sens_data10, sens_data11, sens_data12, sens_data13) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13)
    cur.execute(postgres_insert_query, record_to_insert)
    conn.commit()
    conn.close()


def insert_tiot_table17_total_count(org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13):
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ INSERT INTO tiot_table17 (org_cd,seq,sens_data1, sens_data2, sens_data3, sens_data6, sens_data7, sens_data8, sens_data9, sens_data10, sens_data11, sens_data12, sens_data13) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13)
    cur.execute(postgres_insert_query, record_to_insert)
    conn.commit()
    conn.close()




def read_last_16():
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ SELECT seq FROM tiot_table16 ORDER BY seq DESC LIMIT 1"""
    cur.execute(postgres_insert_query)
    result = cur.fetchall()
    conn.close()
    return result

def read_last_17():
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ SELECT seq FROM tiot_table17 ORDER BY seq DESC LIMIT 1"""
    cur.execute(postgres_insert_query)
    result = cur.fetchall()
    conn.close()
    return result


def capture(cameraid,Year_Month_Date,Hour_Minute):
    reader = easyocr.Reader(['en'],gpu = True,recog_network ='english_g2')
    if cameraid==1:
        ip='192.168.123.15'
    elif cameraid==2:
        ip='192.168.123.16'
    else:
        print('error')
        pass
    path=f'./Camera/{Year_Month_Date}'
    os.makedirs(path,exist_ok=True)
    filename=str(cameraid)+'_'+str(Hour_Minute)
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
            now2=datetime.datetime.now()
            Sec=now2.strftime('%S')
            cv2.imwrite(f"{path}/{filename}_{Sec}.jpg" , vidframe)
            vidcap.release()
            a=cv2.imread(f"{path}/{filename}_{Sec}.jpg")
            # print(a)
            # a1=a[407:441,637:741]
            # read1 = reader.readtext(a1,allowlist=['1','2','3','4','5','6','7','8','9','0'])
            # print(read1)
            if cameraid==1:
                # img_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                # template = cv2.imread('1_template.jpg',0)
                # w, h = template.shape[::-1]

                # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
                # print(f'{cameraid} {np.max(res)}')
                # if np.max(res)>0.4:
                if True:
                    a1=a[407:441,637:741]
                    a2=a[385:432,902:1035]
                    a3=a[695:756,2048:2225]
                    a4=a[944:1002,2034:2205]
                    a5=a[1175:1239,2017:2184]
                    a6=a[1400:1458,1995:2153]
                    a7=a[372:424,1372:1526]
                    
                    try:
                        read1 = reader.readtext(a1,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read2 = reader.readtext(a2,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read3 = reader.readtext(a3,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read4 = reader.readtext(a4,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read5 = reader.readtext(a5,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read6 = reader.readtext(a6,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read7 = reader.readtext(a7,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        # print(read1)
                        total_count=read1[0][1]
                        crack_total=read2[0][1]
                        vertical_crack=read3[0][1]
                        horizontal_crack=read4[0][1]
                        shoulder_crack=read5[0][1]
                        heel_crack=read6[0][1]
                        total_count2=read7[0][1]
                        
                        
                        total_ng_rate=round(int(crack_total)/int(total_count),6)
                        vertical_crack_rate=round(int(vertical_crack)/int(total_count),6)
                        horizontal_crack_rate=round(int(horizontal_crack)/int(total_count),6)
                        shoulder_crack_rate=round(int(shoulder_crack)/int(total_count),6)
                        heel_crack_rate=round(int(heel_crack)/int(total_count),6)
                        org_cd='010101'
                        # print(org_cd,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                        try:
                            if total_ng_rate>1 or vertical_crack_rate>1 or horizontal_crack_rate>1 or shoulder_crack_rate>1 or heel_crack_rate>1 or int(total_count)<int(crack_total) or int(crack_total)>int(vertical_crack)+int(horizontal_crack)+int(shoulder_crack)+int(heel_crack) or int(total_count)!=int(total_count2):
                                pass
                            else:
                                try:
                                    seq=read_last_16()[0][0]
                                    seq=seq+1
                                except:
                                    seq= 1
                            
                                try:
                                    Itemcode=None
                                    # print(org_cd,seq,Itemcode,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                                    insert_tiot_table16_total_count(org_cd,seq,Itemcode,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                                except:
                                    pass
                        except:
                            pass
                    except:
                        pass
                        
                    
                
            elif cameraid==2:
                # img_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                # template = cv2.imread('2_template.jpg',0)
                # w, h = template.shape[::-1]

                # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
                # print(f'{cameraid} {np.max(res)}')
                # if np.max(res)>0.4:
                if True:
                    a1=a[636:677,1230:1343]
                    a2=a[621:665,1507:1644]
                    a3=a[878:923,1711:1861]
                    a4=a[1106:1157,1720:1866]
                    a5=a[1317:1374,1728:1874]
                    a6=a[1520:1575,1736:1877]
                    a7=a[613:668,1986:2152]
                    
                    try:
                        read1 = reader.readtext(a1,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read2 = reader.readtext(a2,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read3 = reader.readtext(a3,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read4 = reader.readtext(a4,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read5 = reader.readtext(a5,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read6 = reader.readtext(a6,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read7 = reader.readtext(a7,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        total_count=read1[0][1]
                        crack_total=read2[0][1]
                        vertical_crack=read3[0][1]
                        horizontal_crack=read4[0][1]
                        shoulder_crack=read5[0][1]
                        heel_crack=read6[0][1]
                        total_count2=read7[0][1]
                        
                        
                        total_ng_rate=round(int(crack_total)/int(total_count),6)
                        vertical_crack_rate=round(int(vertical_crack)/int(total_count),6)
                        horizontal_crack_rate=round(int(horizontal_crack)/int(total_count),6)
                        shoulder_crack_rate=round(int(shoulder_crack)/int(total_count),6)
                        heel_crack_rate=round(int(heel_crack)/int(total_count),6)
                        org_cd='010101'
                        print(org_cd,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                        try:
                            if total_ng_rate>1 or vertical_crack_rate>1 or horizontal_crack_rate>1 or shoulder_crack_rate>1 or heel_crack_rate>1 or int(total_count)<int(crack_total) or int(crack_total)>int(vertical_crack)+int(horizontal_crack)+int(shoulder_crack)+int(heel_crack) or int(total_count)!=int(total_count2):
                                pass
                            else:
                                try:
                                    seq=read_last_17()[0][0]
                                    seq=seq+1
                                except:
                                    seq= 1
                            
                                try:
                                    Itemcode=None
                                    # print(org_cd,seq,Itemcode,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                                    insert_tiot_table17_total_count(org_cd,seq,Itemcode,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                                except:
                                    pass
                        except:
                            pass
                    except:
                        pass
        suc,vidframe = vidcap.read()
        count += 1
    


    
        ################################################################판독 후 data 넣어주기
    




def main():
    m3=multiprocessing.Process(target=delete)
    m3.start()
    
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
            # time.sleep(60)
        except:
            continue
    

def delete():
    while True:
        time.sleep(40000)
        try:
            shutil.rmtree('./Camera')
        except:
            continue

if __name__== '__main__':
    # multiprocessing.set_start_method('spawn')
    main()
