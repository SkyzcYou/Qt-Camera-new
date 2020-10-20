import face_recognition
import os,re
from numpy import *


def load_face_data():
    '''
    加载本地人脸库并编码
    known_face_encodings ： 返回一个人脸编码数组，作为已知识别库
    known_face_names ： 返回一个人脸姓名数组，元素为图片文件名
    :return:
    known_list_db (list) : 列表有两个元素，0是已知人脸的编码数组，1是姓名
    '''
    # 已知人脸库 Start
    known_list_db = []

    known_face_encodings = []
    known_face_names = []

    # 循环遍历文件夹获取文件名
    for root,dirs,files in os.walk(r"/home/pi/PyProject/awesome-cool/know_face_img"):
        for file in files:
            fold_name = root
            img_file_path = os.path.join(root,file)
            img_file_name = re.findall(r'know_face_img/(.*?)-.*.jpg',img_file_path)[0] # 正则获取文件名作为姓名

            print(fold_name)
            print(img_file_path)
            print(img_file_name)

            try:
                # 编码 加入 list
                image = face_recognition.load_image_file(img_file_path)
                known_face_encodings.append(face_recognition.face_encodings(image)[0])
                # 写入姓名
                known_face_names.append(img_file_name)
            except:
                continue


    print("======^_^本地人脸数据库加载完毕^_^======")

    known_list_db.append(known_face_encodings)
    known_list_db.append(known_face_names)

    # known_list_db[] 列表有两个元素，0是已知人脸的编码数组，1是姓名
    # print(known_list_db[0])
    # print(known_list_db[1])
    return known_list_db





