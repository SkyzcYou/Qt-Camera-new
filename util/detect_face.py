import face_recognition
import cv2
import numpy as np



def get_detect_face(rgb_frame, known_list_db):
    """
    人脸检测及人脸识别
    检测图片中的人脸位置、识别人脸数据
    Args:
        rgb_frame (numpy.ndarray): 传入一个经过 RGB 处理后的 OpenCV 格式的图像
        known_list_db (list) 传入加载好的已知数据库,包含编码库和姓名库
    Returns:
        name : 返回图片中人物的姓名
    """

    # 创建已知人脸编码及其名称的数组
    known_face_encodings = known_list_db[0]
    known_face_names = known_list_db[1]

    # print(known_face_encodings)
    # print(known_face_names)
    # print("已知人脸库加载成功==")



    # 初始化变量
    face_locations = []
    face_encodings = []
    face_names = []

    # # 将视频帧调整为1/4大小，以便更快地进行人脸识别处理
    rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
    # # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
    rgb_frame = rgb_frame[:, :, ::-1]
    # 对传入的 RGB 处理后的 OpenCV 格式的图像进行编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # 查看该面是否与已知面匹配
        # print(known_face_encodings)
        # print(known_face_encodings[0])
        # print(face_encoding)
        # print(type(known_face_encodings))
        # print(type(face_encoding))
        # print(type(known_face_encodings[0]))

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"

        # 如果在已知的面编码中找到匹配项，只需使用第一个。
        # 如果匹配项中为True：
        # 第一个匹配索引=匹配.索引（正确）
        # 名称=已知的面名称[第一个匹配的索引]

        # 或者，使用与新面的距离最小的已知面
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        print(name)
        return name