import numpy as np
import pypinyin
import time
import cv2
import os

def save_image(name_str,face_image):
    """
    保存人脸图片到本地,持久化处理:
    传入姓名和人脸编码，保存编码到本地数据库(known_people_data/NAME-FACE_ID.npy)
    Args:
    name_str(str) : 要保存的人脸英文名
    face_image() : 要保存的人脸帧 (cv2抓取的视频帧)
    :return:
    """

    # 1. 利用时间戳创建 FACE_ID
    ticks = time.time()
    FACE_ID = str(ticks).replace('.', '')
    # 2. 文件名为 姓名拼音-当前时间戳.jpg
    pre = r"/home/pi/PyProject/awesome-cool/know_face_img/"
    file_name = pre + name_str + '-' + FACE_ID + ".jpg"
    # print(file_name)
    # 3.帧处理
    # 将视频帧调整为1/4大小，以便更快地进行人脸识别处理
    small_frame = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)
    # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
    rgb_small_frame = small_frame[:, :, ::-1]

    # 。4. save file
    print("collection_face_image==> %s", cv2.imwrite(file_name,rgb_small_frame))

# name_str = 'GOLANG'
# video_capture = cv2.VideoCapture(0)
# # 抓取一帧视频
# ret, frame = video_capture.read()
# 将视频帧调整为1/4大小，以便更快地进行人脸识别处理
# small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
# rgb_small_frame = frame[:, :, ::-1]

# 显示结果图像
# cv2.imshow('Video', frame)
# cv2.waitKey(0)


# save_image(name_str,face_image=frame)





