import paddlehub as hub
import cv2

''' 加载训练模型:pyramidbox_lite_server_mask '''
module = hub.Module(name="pyramidbox_lite_server_mask")

def detect_mask(images_list):
    """
    检测人脸是否戴口罩
    Args:
        images_list (list[numpy.ndarray]): 传入一个 List，List 元素类型为 numpy.ndarray
    Returns:
        isMake(Boolean): 返回 Boolean 变量 isMask，Ture 表示戴口罩，False表示未带口罩
    """
    isMask = False
    results = module.face_detection(images=images_list)
    # print(results)
    # print(results[0]['data'][0]['label'])
    label = results[0]['data'][0]['label']
    print("Original label == ",label)
    if label == "MASK":
        isMask = True
        print("detect_mask-->:MASK")
        return isMask
    else:
        print("detect_mask-->:NO MASK")

    return isMask


''' 测试程序段，分别测试一张未戴口罩o和戴口罩的图片x '''
# o = detect_mask([cv2.imread('/home/giftiay/PycharmProjects/awesome-cool/known_people/pony-2398472934.jpg')])
# x = detect_mask([cv2.imread('/home/giftiay/PycharmProjects/awesome-cool/unknown_pictures/hus.jpg')])
# print(o)
# print(x)