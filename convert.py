#directory = 'drive/MyDrive/colab/tensorflow/workspace/images'
#data_directory = 'drive/MyDrive/colab/tensorflow/workspace/data/malaria/'
#class_weights = '{ "red blood cell": 1.0, "trophozoite": 52.559402579769184, "difficult": 175.55555555555557, "ring": 219.3201133144476, "schizont": 432.513966480447, "gametocyte": 537.6388888888889, "leukocyte": 751.6504854368932, "infected": 15.4459459459, "uninfected": 0.5167269439}'
#class_weights = json.dumps(class_weights)
#class_map = '{ "red blood cell": "uninfected", "trophozoite": "infected", "difficult": "infected", "ring": "infected", "schizont": "infected", "gametocyte": "infected", "leukocyte": "rmv"}'
#class_map = json.dumps(class_map)
#!python drive/MyDrive/colab/tensorflow/scripts/convert.py -id {directory} -dd {data_directory} -cm {class_map} -cw {class_weights}

import json
from PIL import Image
import os
import shutil
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Script to convert Malaria Dataset annotation to pascal voc")
parser.add_argument("-id",
                    "--img_dir",
                    help="Path to the folder where the the img files and annotations are stored.",
                    type=str)
parser.add_argument("-dd",
                    "--data_dir",
                    help="Path to the folder where the the compressed files are stored.", type=str)
parser.add_argument("-cw",
                    "--cls_wts",
                    help="a string containing the class weights.", type=str)
parser.add_argument("-iu",
                    "--cls_inf",
                    help="the infection status of the dataset",
                    type=str, default=None)
parser.add_argument("-cm",
                    "--cls_mp",
                    help="a .json file containing classes to be mapped",
                    type=str, default=None)

args = parser.parse_args()

directory = args.img_dir
data_directory = args.data_dir
data_infection = args.cls_inf
class_weights = json.loads(args.cls_wts)
class_maps = json.loads(args.cls_mp)

#print (directory)
#print (data_directory)
#print (class_weights)
#print (class_maps)

try:
    #os.mkdir(directory+'/labels/')
    os.mkdir(directory)
    print(directory + ' folder has been successfully created')
except OSError as error:
    print(error)

try:
    path = os.path.join(directory, 'training', 'uninfected')
    os.makedirs(path)
    print(path + ' folder has been successfully created')
except OSError as error:
    print(error)

try:
    path = os.path.join(directory, 'training', 'infected')
    os.makedirs(path)
    print(path + ' folder has been successfully created')
except OSError as error:
    print(error)

try:
    os.mkdir(directory + '/test/')
    print(directory + '/test/' + ' has been successfully created')
except OSError as error:
    print(error)

def wt_frm_name(name):
  wt = class_weights[name]
  return wt

def map_frm_name(name):
  wt = class_maps[name]
  return wt

for mode in ['training', 'test']:
    inf_ct = 0
    uninf_ct = 0
    rmv = 0
    no_of_files = 0
    with open (data_directory + "/{}.json".format(mode), "r") as myfile:
        data = json.loads(myfile.read())
        for status in ['uninfected', 'infected']:
            for sample in data:
                inf_ct_int = 0
                uninf_ct_int = 0 # count for each image
                image_src = data_directory + sample['image']['pathname']
                #img_dir = directory + '/labels/{}'.format(mode)
                if mode == 'training':
                    img_dir = directory + '/{}/{}'.format(mode, status)
                    #print(img_dir)
                else:
                    img_dir = directory + '/' + '{}'.format(mode)
                image = Image.open(image_src)
                width, height = image.size

                no_of_files += 1 # count the number of images

                #print(image_src)
                #print(img_dir + sample['image']['pathname'])

                filename = sample['image']['pathname'].split('/')[-1]

                if mode == 'training':
                    path = '..' + sample['image']['pathname'] # todo correct pathname directory
                    #print(sample['image']['pathname'])
                else:
                    path = '..' + sample['image']['pathname']

                output = "<annotation>\n\t<folder>malaria</folder>"
                output += "\n\t<filename>{}</filename>\n\t<path>{}</path>".format(filename, path)
                output += "\n\t<size>\n\t\t<width>{}</width>\n\t\t<height>{}</height>".format(width, height)
                output += "\n\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>"

                for object in sample['objects']:
                    category = map_frm_name(object['category'])
                    #print(category)
                    xmin = object['bounding_box']['minimum']['c']
                    ymin = object['bounding_box']['minimum']['r']
                    xmax = object['bounding_box']['maximum']['c']
                    ymax = object['bounding_box']['maximum']['r']

                    #https://stackoverflow.com/questions/51862997/class-weights-for-balancing-data-in-tensorflow-object-detection-api
                    if category == 'infected': 
                      inf_ct += 1
                      inf_ct_int += 1
                    if category == 'uninfected':
                      uninf_ct += 1
                      uninf_ct_int += 1
                    if category == 'rmv':
                      rmv += 1
                    if category != 'rmv':
                        if mode == 'training':
                            if category == status: # take out rbcs from annotation

                              output += "\n\t<object>\n\t\t<name>{}</name>\n\t\t<pose>Unspecified</pose>".format(category)
                              output += "\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>"
                              output += "\n\t\t\t<xmin>{}</xmin>\n\t\t\t<ymin>{}</ymin>".format(xmin, ymin)
                              output += "\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>".format(xmax, ymax)
                              output += "\n\t\t</bndbox>"
                              output += "\n\t\t<weight>{}</weight>".format(wt_frm_name(category)) # to add class weights to annotation https://stackoverflow.com/questions/51862997/class-weights-for-balancing-data-in-tensorflow-object-detection-api
                              output += "\n\t</object>"
                            
                         else:
                            output += "\n\t<object>\n\t\t<name>{}</name>\n\t\t<pose>Unspecified</pose>".format(category)
                            output += "\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>"
                            output += "\n\t\t\t<xmin>{}</xmin>\n\t\t\t<ymin>{}</ymin>".format(xmin, ymin)
                            output += "\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>".format(xmax, ymax)
                            output += "\n\t\t</bndbox>"
                            output += "\n\t\t<weight>{}</weight>".format(wt_frm_name(category)) # to add class weights to annotation https://stackoverflow.com/questions/51862997/class-weights-for-balancing-data-in-tensorflow-object-detection-api
                            output += "\n\t</object>"

                output += "\n</annotation>"

                if mode == 'training':
                    if inf_ct_int >> 0 or uninf_ct_int >> 0:
                        shutil.copyfile(image_src, img_dir + '/'+ sample['image']['pathname'].split('/')[-1])
                        with open(directory + "/{}/{}/{}.xml".format(mode, status, sample['image']['pathname'].split('/')[-1].split('.')[0]), "w") as myfile:
                            myfile.write(output)
                else:
                    shutil.copyfile(image_src, img_dir + '/'+ sample['image']['pathname'].split('/')[-1])
                    with open(directory + "/{}/{}.xml".format(mode, sample['image']['pathname'].split('/')[-1].split('.')[0]), "w") as myfile:
                        myfile.write(output)

    print ("{} dataset created with {} files containing {} uninfected cells, {} infected cells. {} cells were removed.".format(mode, no_of_files, uninf_ct, inf_ct, rmv))
