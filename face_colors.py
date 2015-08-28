import cv2 as opencv
from PIL import Image
import numpy as np
import colorsys
import math
import sys

imgs = sys.argv[1:]

face_cascade = opencv.CascadeClassifier("haarcascade_frontalface_alt.xml")

r = 0
g = 0
b = 0
n_faces = 0

gray_mean = 0

for image in imgs:
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

    face_mean = opencv.mean(gray_face)[0]
    gray_mean += face_mean

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
    rgb = colorsys.hsv_to_rgb(color['center'][0]/180.0,
                              color['center'][1]/255.0, color['center'][2]/255.0)
    rgb = [x * 256 for x in rgb]

    n_faces = n_faces + 1
    r = r + rgb[0]
    g = g + rgb[1]
    b = b + rgb[2]


r = int(r/n_faces)
g = int(g/n_faces)
b = int(b/n_faces)
gray_mean = gray_mean / n_faces

rgb_color = "rgb({r},{g},{b})".format(r=r, g=g, b=b)
answer = 'YES'  # constant because, like this is ever going to change

TEMPLATE = """
<html>
<head>
<title>Is Burning Man Still White?</title>
</head>

<body style="background-color: {color}">
<center>
<h1 style="font-size: 20em">{answer}</h1>

<p><a href="http://burningman.org/network/about-us/people/board-of-directors/">See for yourself.</a></p>
<p><a href="https://github.com/zedshaw/isburningmanstillwhite">Fork us on github.</a></p>
</center>
</body>
</html>
""".format(color=rgb_color, answer=answer)

output = open("index.html", 'w')
output.write(TEMPLATE)

