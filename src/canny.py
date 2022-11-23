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

contours, _ = cv2.findContours(closed_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)
print(type(contours))
print(len(contours))
print(contours[0].shape)


def closedContours(src: np.ndarray, threshold1: int, threshold2: int , kernel_size: int) -> tuple:
	"""
	Take a source image
	Finds the image edges using the Canny algorithm
	Closes edge gaps with morphology close operations
	Finds all object external contours
	:Return: a tuple of contours
	"""
	assert type(src)==np.ndarray
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	edges = cv2.Canny(src, threshold1, threshold2)
	closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	cnt, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return contours


cnt = closedContours(img, 600, 150, 5)
print(len(cnt))

cont_img = cv2.drawContours(img, cnt, -1, (255, 0, 0), 3)

showImage(cont_img)