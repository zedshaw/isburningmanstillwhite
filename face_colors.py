import argparse
import cv2 as opencv
from PIL import Image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("imgs", help="Path to the image(s) in which to find faces.", nargs="+")
parser.add_argument("--output", help="Path to json output.", default="STDOUT")
args = parser.parse_args()

face_cascade = opencv.CascadeClassifier("haarcascade_frontalface_alt.xml")

for image in args.imgs:
  color = opencv.imread(image)
  gray = opencv.cvtColor(color, opencv.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)


  for(x,y,w,h) in faces:
    gray_face = gray[y:(y+h), x:(x+w)]
    color_face = color[y:(y+h), x:(x+w)]

    Image.fromarray(gray_face).save("{}/{}".format("small-gray",image.rsplit("/",1)[1]))
    Image.fromarray(color_face).save("{}/{}".format("small-color",image.rsplit("/",1)[1]))
