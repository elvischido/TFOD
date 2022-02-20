import json
from PIL import Image
import os
import shutil

#directory = '/content/Tensorflow/workspace/images/malaria'
directory = '/content/Tensorflow/workspace/images'
data_directory = '/content/Tensorflow/workspace/data/malaria/'

try:
    #os.mkdir(directory+'/labels/')
    os.mkdir(directory)
except OSError as error:
    print(error)

try:
    os.mkdir(directory +'/training/')
except OSError as error:
    print(error)

try:
    #os.mkdir(directory + '/labels/test/')
    os.mkdir(directory + '/test/')
except OSError as error:
    print(error)

def wt_frm_name(name):
  # Define the class names and their weight
  #class_names = [' ','red blood cell', 'trophozoite', 'difficult', 'ring', 'schizont', 'gametocyte', 'leukocyte']
  #class_weights = [0, 1.0, 52.559402579769184, 175.55555555555557, 219.3201133144476, 432.513966480447, 537.6388888888889, 751.6504854368932]
  class_weights = {'red blood cell': 1.0, 'trophozoite': 52.559402579769184, 'difficult': 175.55555555555557, 'ring': 219.3201133144476, 'schizont': 432.513966480447, 'gametocyte': 537.6388888888889, 'leukocyte': 751.6504854368932}
  #class_weights = [{1: 1.0}, {2: 52.559402579769184}, {3: 175.55555555555557}, {4: 219.3201133144476}, {5: 432.513966480447}, {6: 537.6388888888889}, {7: 751.6504854368932}]

  wt = class_weights[name]
  #print(class_weights[name])
  return wt

for mode in ['training', 'test']:
    with open (data_directory + "/{}.json".format(mode), "r") as myfile:
        data = json.loads(myfile.read())

        for sample in data:
            image_src = data_directory + sample['image']['pathname']
            #img_dir = directory + '/labels/{}'.format(mode)
            img_dir = directory + '/' + '{}'.format(mode)
            image = Image.open(image_src)
            width, height = image.size

            #print(image_src)
            #print(img_dir + sample['image']['pathname'])
            
            shutil.copyfile(image_src, img_dir + '/' + sample['image']['pathname'].split('/')[-1])


            filename = sample['image']['pathname'].split('/')[-1]
            path = '..' + sample['image']['pathname']

            output = "<annotation>\n\t<folder>malaria</folder>"
            output += "\n\t<filename>{}</filename>\n\t<path>{}</path>".format(filename, path)
            output += "\n\t<size>\n\t\t<width>{}</width>\n\t\t<height>{}</height>".format(width, height)
            output += "\n\t\t<depth>3</depth>\n\t</size>\n\t<segmented>0</segmented>"

            for object in sample['objects']:
                category = object['category']
                #print(category)
                xmin = object['bounding_box']['minimum']['c']
                ymin = object['bounding_box']['minimum']['r']
                xmax = object['bounding_box']['maximum']['c']
                ymax = object['bounding_box']['maximum']['r']

                #https://stackoverflow.com/questions/51862997/class-weights-for-balancing-data-in-tensorflow-object-detection-api
                #if category != 'red blood cell': # take out rbcs from annotation
                output += "\n\t<object>\n\t\t<name>{}</name>\n\t\t<pose>Unspecified</pose>".format(category)
               
                output += "\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>"
                output += "\n\t\t\t<xmin>{}</xmin>\n\t\t\t<ymin>{}</ymin>".format(xmin, ymin)
                output += "\n\t\t\t<xmax>{}</xmax>\n\t\t\t<ymax>{}</ymax>".format(xmax, ymax)
                output += "\n\t\t</bndbox>"
                output += "\n\t\t<weight>{}</weight>".format(wt_frm_name(category)) # to add class weights to annotation https://stackoverflow.com/questions/51862997/class-weights-for-balancing-data-in-tensorflow-object-detection-api
                output += "\n\t</object>"

            output += "\n</annotation>"

            with open(directory + "/{}/{}.xml".format(mode, sample['image']['pathname'].split('/')[-1].split('.')[0]), "w") as myfile:
                myfile.write(output)
