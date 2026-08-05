import cv2

image_color_array = cv2.imread('input.jpg')
image_gray_array = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imwrite('output_color.jpg', image_color_array)
cv2.imwrite('output_gray.jpg', image_gray_array)