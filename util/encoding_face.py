import face_recognition
import cv2
import numpy as np

def get_coding_img(rgb_small_frame):
    """
    传入一张RGB图片进行编码
    保存一个已知照片的编码数组到本地 txt 文件，测试能否在比较时成功
    Args:
        images_list (list[numpy.ndarray]): 传入一个 List，List 元素类型为 numpy.ndarray
    Returns:
        isMake(Boolean): 返回 Boolean 变量 isMask，Ture 表示戴口罩，False表示未带口罩
    """
    # encoding
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]

    return face_encodings
def get_local_img(img_path):
    """
    传入一张本地图片进行编码
    保存一个已知照片的编码数组到本地 txt 文件，测试能否在比较时成功
    Args:
        images_list (list[numpy.ndarray]): 传入一个 List，List 元素类型为 numpy.ndarray
    Returns:
        isMake(Boolean): 返回 Boolean 变量 isMask，Ture 表示戴口罩，False表示未带口罩
    """
    # encoding
    face_locations = face_recognition.load_image_file(img_path)
    face_encodings = face_recognition.face_encodings(face_locations)[0]

    return face_encodings
