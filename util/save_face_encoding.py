import numpy as np
import pypinyin
import sqlite3
import time
import os

def save_encoding(name_str,face_encode):
    """
    保存人脸编码到本地,持久化处理:
    传入姓名和人脸编码，保存编码到本地数据库(known_people_data/NAME-FACE_ID.npy)
    Args:
    name_str(str) : 要保存的中文姓名字符串
    face_encode(numpy.array) : 要保存的人脸编码
    :return:
    """

    # 1. 获取姓名拼音
    NAME_PinYin = name_to_pinyin(name_str)
    # 2. 利用时间戳创建 FACE_ID
    ticks = time.time()
    FACE_ID = str(ticks).replace('.', '')
    # 3. 文件名为 姓名拼音-当前时间戳.npy
    file_name = NAME_PinYin + '-' + FACE_ID + ".npy"
    # save file
    np.save(file_name, face_encode)

    # save info to db.
    # 链接人员信息数据库创建 name,file_name
    # connect = sqlite3.connect('/home/giftiay/PycharmProjects/awesome-cool/db/awesome.db')
    # cursor = connect.cursor()
    #
    # inset = "INSERT INTO people_info values('{}','{}')".format(name_str, "file_name")
    # cursor.execute(inset)
    # connect.commit()
    # # 关闭游标
    # cursor.close()
    # # 断开数据库连接
    # cursor.close()


def name_to_pinyin(name_str):
    """
    中文姓名转拼音
    :return: name_pinyin(str)
    """
    name_pinyin = ''
    for i in pypinyin.pinyin(name_str,style=pypinyin.NORMAL):
        name_pinyin += " ".join(i)
    return name_pinyin

# Test name_to_pinyin()
# print(name_to_pinyin("游正材"))






