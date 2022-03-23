import os
import os.path
import torchvision
from PIL import Image
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]

    train_left_img=[]
    train_right_img=[]
    train_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []
    image=filepath+'/'+image[0]
    disp=filepath+'/'+disp[0]

    monkaa_path=os.listdir(image)
    length=len(monkaa_path)
    i=0
    for docu in monkaa_path:
        i=i+1
        for im in os.listdir(image+'/'+docu+'/left/'):
            if is_img_file(image+'/'+docu+'/left/'+im) and i<0.77*length:
                train_left_img.append(image+'/'+docu+'/left/'+im)
                train_right_img.append(image+'/'+docu+'/right/'+im)
                train_left_disp.append(disp+'/'+docu+'/left/'+im.split('.')[0]+'.pfm')
            if is_img_file(image+'/'+docu+'/left/'+im) and i==length//10:
                test_left_img.append(image+'/'+docu+'/left/'+im)
                test_right_img.append(image+'/'+docu+'/right/'+im)
                test_left_disp.append(disp+'/'+docu+'/left/'+im.split('.')[0]+'.pfm')
                break


    return train_left_img,train_right_img,train_left_disp,test_left_img,test_right_img,test_left_disp

