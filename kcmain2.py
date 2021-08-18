import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse

import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


import easyocr
import psycopg2
import threading,os
import multiprocessing
import shutil



################################################################################################################################################################################
def insert_tiot_table18_total_count(org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13):
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ INSERT INTO tiot_table18 (org_cd,seq,sens_data1, sens_data2, sens_data3, sens_data6, sens_data7, sens_data8, sens_data9, sens_data10, sens_data11, sens_data12, sens_data13) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13)
    cur.execute(postgres_insert_query, record_to_insert)
    conn.commit()
    conn.close()


def insert_tiot_table19_total_count(org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13):
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ INSERT INTO tiot_table19 (org_cd,seq,sens_data1, sens_data2, sens_data3, sens_data6, sens_data7, sens_data8, sens_data9, sens_data10, sens_data11, sens_data12, sens_data13) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (org_cd,seq,sens_data1,sens_data2,sens_data3,sens_data6,sens_data7,sens_data8,sens_data9,sens_data10,sens_data11,sens_data12,sens_data13)
    cur.execute(postgres_insert_query, record_to_insert)
    conn.commit()
    conn.close()




def read_last_18():
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ SELECT seq FROM tiot_table18 ORDER BY seq DESC LIMIT 1"""
    cur.execute(postgres_insert_query)
    result = cur.fetchall()
    conn.close()
    return result

def read_last_19():
    conn_string="host = '192.168.123.177' dbname = 'edgedb' user='postgres' password='Kcg2021!' "
    conn = psycopg2.connect(conn_string,options='-c search_path=iot')
    cur = conn.cursor()
    postgres_insert_query = """ SELECT seq FROM tiot_table19 ORDER BY seq DESC LIMIT 1"""
    cur.execute(postgres_insert_query)
    result = cur.fetchall()
    conn.close()
    return result
################################################################################################################################################################################


def capture(ip):
    if ip.endswith('6'):
        reader = easyocr.Reader(['en'],gpu = True,recog_network ='english_g2')
        device = select_device('0')
        

        half = device.type != 'cpu'  # half precision only supported on CUDA
        
        weights='/home/qfactory/Camera/best.pt'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(640, s=model.stride.max())  # check img_size
        if half:
            model.half()
        
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    n=0
    while True:
        n+=1
        cap=cv2.VideoCapture(f"rtsp://admin:bizarim0@{ip}:554/Streaming/channels/101")
        # print(cap.isOpened())
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        
        if cap.isOpened():
            
            ret,frame=cap.read()
            cap.release()####################################################
            if ret:
                ##########################################################################################
                path=f'/home/qfactory/Camera/tmp/'
                out='/home/qfactory/Camera/out'
                
                if ip.endswith('6'):
                    
                    os.makedirs(path,exist_ok=True)
                    cv2.imwrite(f'{path}/2_{time.ctime()}.jpg',frame)
                    while True:
                        if len(os.listdir(path))==2:
                            t1= threading.Thread(target=detect, args=[path,out,model,names,colors,imgsz,half,device,reader])
                            t1.start()
                            t1.join()
                            shutil.rmtree(path,ignore_errors=True)
                            break
                        else:
                            time.sleep(0.2)
                            continue
                elif ip.endswith('5'):
                    try:
                        if len(os.listdir(path))==1:
                            cv2.imwrite(f'{path}/1_{time.ctime()}.jpg',frame)
                    except:
                        pass
            else:
                continue



def detect(source,out,model,names,colors,imgsz,half,device,reader):
    print('detect시작\n')
    os.makedirs(out,exist_ok=True)  # make new output folder

    dataset = LoadImages(source, img_size=imgsz)


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.2, classes=None, agnostic=None)
        # t2 = time_synchronized()

        # Apply Classifier


        # Process detections
        
        for i, det in enumerate(pred):  # detections per image
            xyxylist=[]
            p, s, im0 = path, '', im0s

            # save_path = str(Path(out) / Path(p).name)
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                
                for *xyxy, conf, cls in det:
                    xyxylist.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                    
                    # if True:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # with open(txt_path, 'a') as f:
                            # f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                            # f.write(('%g ' * 5 + '\n') % (cls, *xyxy))

                    # if save_img or view_img:  # Add bbox to image
                    label = '%s' % (names[int(cls)])
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
            # cv2.imwrite(f'./detect/{Path(p).name}',im0)
            if Path(p).name.startswith('1'):
                if len(xyxylist)==7:
                    first3box=sorted(xyxylist)[:3]
                    remainbox=sorted(sorted(xyxylist)[3:],key=lambda x:x[1])
                    # print(first3box[0])
                    try:
                        a1=im0[first3box[0][1]:first3box[0][3],first3box[0][0]+85:first3box[0][2]]
                        a2=im0[first3box[1][1]:first3box[1][3],first3box[1][0]+100:first3box[1][2]]
                        a3=im0[first3box[2][1]:first3box[2][3],first3box[2][0]+130:first3box[2][2]]
                        a4=im0[remainbox[0][1]+56:remainbox[0][3],remainbox[0][0]:remainbox[0][2]]
                        a5=im0[remainbox[1][1]+56:remainbox[1][3],remainbox[1][0]:remainbox[1][2]]
                        a6=im0[remainbox[2][1]+56:remainbox[2][3],remainbox[2][0]:remainbox[2][2]]
                        a7=im0[remainbox[3][1]+56:remainbox[3][3],remainbox[3][0]:remainbox[3][2]]
                        read1 = reader.readtext(a1,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read2 = reader.readtext(a2,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read3 = reader.readtext(a3,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read4 = reader.readtext(a4,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read5 = reader.readtext(a5,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read6 = reader.readtext(a6,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        read7 = reader.readtext(a7,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                        print('1\n')
                        print(f'{read1[0][1]}_________{read1[0][-1]}')
                        print(f'{read2[0][1]}_________{read2[0][-1]}')
                        print(f'{read3[0][1]}_________{read3[0][-1]}')
                        print(f'{read4[0][1]}_________{read4[0][-1]}')
                        print(f'{read5[0][1]}_________{read5[0][-1]}')
                        print(f'{read6[0][1]}_________{read6[0][-1]}')
                        print(f'{read7[0][1]}_________{read7[0][-1]}')

                        
                    ############################################################################################################
                        if read1[0][-1]>0.8 and read2[0][-1]>0.8 and read3[0][-1]>0.8 and read4[0][-1]>0.8 and read5[0][-1]>0.8 and read6[0][-1]>0.8 and read7[0][-1]>0.8:
                            
                            total_count=read1[0][1]
                            crack_total=read2[0][1]
                            vertical_crack=read4[0][1]
                            horizontal_crack=read5[0][1]
                            shoulder_crack=read6[0][1]
                            heel_crack=read7[0][1]
                            total_count2=read3[0][1]
                            total_ng_rate=round(int(crack_total)/int(total_count),6)
                            vertical_crack_rate=round(int(vertical_crack)/int(total_count),6)
                            horizontal_crack_rate=round(int(horizontal_crack)/int(total_count),6)
                            shoulder_crack_rate=round(int(shoulder_crack)/int(total_count),6)
                            heel_crack_rate=round(int(heel_crack)/int(total_count),6)
                            org_cd='010101'
                            if len(total_count)==7 and len(crack_total)==7 and len(vertical_crack)==5 and len(horizontal_crack)==5 and len(shoulder_crack)==5 and len(heel_crack)==5:
                                try:
                                    if total_ng_rate>1 or vertical_crack_rate>1 or horizontal_crack_rate>1 or shoulder_crack_rate>1 or heel_crack_rate>1 or int(total_count)<int(crack_total) or int(crack_total)>int(vertical_crack)+int(horizontal_crack)+int(shoulder_crack)+int(heel_crack) or int(total_count)!=int(total_count2):
                                        pass
                                    else:
                                        try:
                                            seq=read_last_18()[0][0]
                                            seq=seq+1
                                        except:
                                            seq= 1
                                        try:
                                            Itemcode=None
                                            insert_tiot_table18_total_count(org_cd,seq,Itemcode,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                                        except:
                                            print(f'dberror{Path(p).name}')
                                            pass
                                except:
                                    pass
                    except:
                        pass


            if Path(p).name.startswith('2'):
                if len(xyxylist)==7:
                    xyxylist=sorted(xyxylist)
                    first2box=xyxylist[:2]
                    lastbox=xyxylist[-1]
                    remainbox=sorted(xyxylist[2:-1],key=lambda x:x[1])
                    a1=im0[first2box[0][1]:first2box[0][3],first2box[0][0]:first2box[0][2]]
                    a2=im0[first2box[1][1]:first2box[1][3],first2box[1][0]+100:first2box[1][2]]
                    a3=im0[remainbox[0][1]+50:remainbox[0][3],remainbox[0][0]:remainbox[0][2]]
                    a4=im0[remainbox[1][1]+50:remainbox[1][3],remainbox[1][0]:remainbox[1][2]]
                    a5=im0[remainbox[2][1]+50:remainbox[2][3],remainbox[2][0]:remainbox[2][2]]
                    a6=im0[remainbox[3][1]+50:remainbox[3][3],remainbox[3][0]:remainbox[3][2]]
                    a7=im0[lastbox[1]:lastbox[3],lastbox[0]:lastbox[2]]
                    read1 = reader.readtext(a1,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    read2 = reader.readtext(a2,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    read3 = reader.readtext(a3,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    read4 = reader.readtext(a4,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    read5 = reader.readtext(a5,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    read6 = reader.readtext(a6,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    read7 = reader.readtext(a7,allowlist=['1','2','3','4','5','6','7','8','9','0'])
                    print('2\n')
                    print(f'{read1[0][1]}_________{read1[0][-1]}')
                    print(f'{read2[0][1]}_________{read2[0][-1]}')
                    print(f'{read3[0][1]}_________{read3[0][-1]}')
                    print(f'{read4[0][1]}_________{read4[0][-1]}')
                    print(f'{read5[0][1]}_________{read5[0][-1]}')
                    print(f'{read6[0][1]}_________{read6[0][-1]}')
                    print(f'{read7[0][1]}_________{read7[0][-1]}')
                    # cv2.imwrite(f'./detect/{Path().name}',im0)
                    #####################################################################################
                    if read1[0][-1]>0.8 and read2[0][-1]>0.8 and read3[0][-1]>0.8 and read4[0][-1]>0.8 and read5[0][-1]>0.8 and read6[0][-1]>0.8 and read7[0][-1]>0.8:
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
                        if len(total_count)==7 and len(crack_total)==7 and len(vertical_crack)==5 and len(horizontal_crack)==5 and len(shoulder_crack)==5 and len(heel_crack)==5:
                            try:
                                if total_ng_rate>1 or vertical_crack_rate>1 or horizontal_crack_rate>1 or shoulder_crack_rate>1 or heel_crack_rate>1 or int(total_count)<int(crack_total) or int(crack_total)>int(vertical_crack)+int(horizontal_crack)+int(shoulder_crack)+int(heel_crack) or int(total_count)!=int(total_count2):
                                    pass
                                else:
                                    try:
                                        seq=read_last_19()[0][0]
                                        seq=seq+1
                                    except:
                                        seq= 1
                                
                                    try:
                                        Itemcode=None
                                        insert_tiot_table19_total_count(org_cd,seq,Itemcode,total_count,total_ng_rate,vertical_crack,vertical_crack_rate,horizontal_crack,horizontal_crack_rate,shoulder_crack,shoulder_crack_rate,heel_crack,heel_crack_rate)
                                        
                                    except:
                                        print(f'dberror{Path(p).name}')
                                        pass
                            except:
                                pass







def reboot():
    time.sleep(3600)
    os.system('reboot')







if __name__ == '__main__':
    try:
        shutil.rmtree('/home/qfactory/Camera/tmp',ignore_errors=True)
        shutil.rmtree('/home/qfactory/Camera/out',ignore_errors=True)
    except:
        pass
    # trigger_comm0 = multiprocessing.Queue()
    t1=multiprocessing.Process(target=capture,args=['192.168.123.15'])
    t1.start()
    t2=multiprocessing.Process(target=capture,args=['192.168.123.16'])
    t2.start()
    t3=multiprocessing.Process(target=reboot)
    t3.start()
