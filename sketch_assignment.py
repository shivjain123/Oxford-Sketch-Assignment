import cv2
from matplotlib import pyplot as plt

image = cv2.imread('mamma.jpg')

"""
cv2.imshow('Mamma', image)
cv2.waitKey(0)
"""

grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""
cv2.imshow('Mamma', grey_image)
cv2.waitKey(0)
"""

inverted_image = 255 - grey_image

"""
cv2.imshow('Mamma', inverted_image)
cv2.waitKey(0)
"""

blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)

"""
cv2.imshow('Mamma', blurred_image)
cv2.waitKey(0)
"""

inverted_blurred_image = 255 - blurred_image

"""
cv2.imshow('Mamma', inverted_blurred_image)
cv2.waitKey(0)
"""

pencil_sketch = cv2.divide(grey_image, inverted_blurred_image, scale=256.0)

cv2.imshow('Mamma', pencil_sketch)
cv2.waitKey(0)