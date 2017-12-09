import cv2
import RSA

def work(gpu_id,img_list):
    rsa = RSA(gpu_id = gpu_id)
    for img_path in img_list:
        img = cv2.imread(img_path)
        rsa.detect(img)
        print('sucess')