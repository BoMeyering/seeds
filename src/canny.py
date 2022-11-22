import cv2
import numpy as np

img_path = '../img/pods_IMG_0003.JPG'
img = cv2.imread(img_path)

print(img.shape)

print(type(img))

cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
cv2.imshow('my_image', img)
cv2.waitKey(0)
cv2.destroyWindow('image')

def showImage(src):
	cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
	cv2.imshow('image', src)
	cv2.waitKey()
	cv2.destroyWindow('image')

print(img[0])