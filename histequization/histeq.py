import cv2
import os

# Specify the filename of the image in the same directory as your script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_filename = os.path.join(script_dir, 'image.jpg')  # Replace with the actual filename of your image

# Read the image
img = cv2.imread(image_filename,0)
hist =cv2.equalizeHist(img)

cv2.imshow('original image',img)
cv2.imshow('Hist',hist)
##cv2.imwrite("hist.jpg", hist)
cv2.waitKey(0)
cv2.destroyAllWindows()
