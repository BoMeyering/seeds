import cv2
import numpy as np

img_path = '../img/pods_IMG_0003.JPG'
img = cv2.imread(img_path)

print(img.shape)

print(type(img))

# cv2.namedWindow('my_image', cv2.WINDOW_NORMAL)
# cv2.imshow('my_image', img)
# cv2.waitKey(0)
# cv2.destroyWindow('image')

def showImage(src):
	cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
	cv2.imshow('image', src)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyWindow('image')



showImage(img)

kernel = np.ones((5, 5), np.uint8)

edges = cv2.Canny(img, 100, 200)
showImage(edges)
print(edges)

closed_edge = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

showImage(closed_edge)

print(edges.shape)