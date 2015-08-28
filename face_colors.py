import argparse
import cv2 as opencv
from PIL import Image
import numpy as np
import colorsys
import math

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("imgs", help="Path to the image(s) in which to find faces.", nargs="+")
parser.add_argument("--output", help="Path to json output.", default="STDOUT")
args = parser.parse_args()

face_cascade = opencv.CascadeClassifier("haarcascade_frontalface_alt.xml")

r = 0
g = 0
b = 0
n_faces = 0

for image in args.imgs:
  color = opencv.imread(image)
  color_rgb = opencv.cvtColor(color, opencv.COLOR_BGR2RGB)
  gray = opencv.cvtColor(color, opencv.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)



  for(x,y,w,h) in faces:
    
    

    opencv.cvtColor(color, opencv.COLOR_BGR2HSV, color)
    gray_face = gray[y:(y+h), x:(x+w)]
    color_face = color[y:(y+h), x:(x+w)]

    # Image.fromarray(gray_face).save("{}/{}".format("small-gray",image.rsplit("/",1)[1]))
    # Image.fromarray(color_face).save("{}/{}".format("small-color",image.rsplit("/",1)[1]))


    ## organize the data for kmeans
    hsv = opencv.split(color_face)
    
    hsv[0] = hsv[0].reshape(hsv[0].shape[0]* hsv[0].shape[1],1)
    hsv[1] = hsv[1].reshape(hsv[1].shape[0]* hsv[1].shape[1],1)
    hsv[2] = hsv[2].reshape(hsv[2].shape[0]* hsv[2].shape[1],1)

    data = opencv.hconcat(hsv)
    ## run kmeans
    criteria = (opencv.TERM_CRITERIA_EPS, 1000, 0)
    compactness,labels,centers =  opencv.kmeans(np.float32(data), 10, criteria, 10, opencv.KMEANS_RANDOM_CENTERS)

    colors = []

    ## sort colors based on size of cluster
    for (i, center) in enumerate(centers):
      labelMask = opencv.inRange(labels,i,i)
      n = opencv.countNonZero(labelMask)
      colors.append({"count" : n, "center": center})

    sortedColors = sorted(colors, key=lambda k: k['count']) 
    sortedColors.reverse() # descending order


    color = sortedColors[0]



    # for (i,color) in enumerate(sortedColors):
      # note: OpenCV uses a hue range of 0-180
    rgb = colorsys.hsv_to_rgb(color['center'][0]/180., color['center'][1]/255., color['center'][2]/255.)
    rgb = [x * 256 for x in rgb]


    n_faces = n_faces + 1
    r = r + rgb[0]
    g = g + rgb[1]
    b = b + rgb[2]


    print("<div style='background-color: rgb({}, {}, {})'><img src='small-color/{}' /></div>".format(int(np.round(rgb[0])),int(np.round(rgb[1])), int(np.round(rgb[2])), image.rsplit("/",1)[1]))


r = r/n_faces
g = g/n_faces
b = b/n_faces

print("<div style='background-color: rgb({}, {}, {})'><p>Average color</p></div>".format(int(r),int(g),int(b)))

