import face_recognition
from PIL import Image, ImageDraw
from resizeimage import resizeimage
import numpy as np
from sklearn.svm import LinearSVC
import glob
import os

# Resize all images in a folder to a certain size
'''
Resize all images
'''
def resize(path, w=1000, h=500):
    for i in [glob.glob(path+'*.%s' % ext) for ext in ["jpg","gif","png","tga","jpeg"]]:
        for item in i:
            with open(item):
                with Image.open(item) as image:
                    print("image, size: ", item, image.size)
                    #resizeimage.resize_cover.validate(image, [w, h], validate=False)
                    try:
                        thumbnail = image.copy()
                        thumbnail.thumbnail((w,h))
                        #contain = resizeimage.resize_contain(image, [w, h])
                        head, tail = os.path.split(item)
                        head, name = os.path.split(head)
                        if not os.path.exists(head+'_resized'):
                            os.makedirs(head+'_resized')
                        if not os.path.exists(head+'_resized/'+name):
                            os.makedirs(head+'_resized/'+name)
                        thumbnail.save(head+'_resized/'+name+'/'+tail)
                    except TypeError as e:
                        print("image not resizable: ", item)
                        print( "Error: %s" % e )


    return

if __name__ == '__main__':
    jitters = 10
    tolerance = 0.5
    threshold = 0
    detection_model = "cnn"
    image_path = "./data/twin_test/Side_By_Side"
    neg_image_path = "./data/Negative/"
    known_image_path = "./data/known/"
    family2_image_path = "./data/Family2/"
    family5_image_path = "./data/Family5/"
    test_image_path = "./data/test/"
    resized_path = "./data/known_resized/"
    
    #for image_path in glob.glob(family2_image_path+'*'):
    #    print("image: ", image_path)
    #    if image_path == []:
    #        continue
    resize(image_path+'/')
