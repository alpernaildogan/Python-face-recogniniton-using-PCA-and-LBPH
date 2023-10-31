import cv2
import os

# Specify the filename of the image in the same directory as your script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_filename = os.path.join(script_dir, 'image.jpg')  # Replace with the actual filename of your image

# Read the image
img = cv2.imread(image_filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

r = 300.0 / gray.shape[1]
dim = (300, int(gray.shape[0] * r))
# resizing of the image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
edges = cv2.Canny(gray,100,200)


cv2.imshow('original image',img)
cv2.imshow('gray image',gray)
cv2.imshow('resized image',resized)
cv2.imshow('edge image',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
