import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.applications import vgg16
from sklearn.neighbors import NearestNeighbors
import cv2


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
img_path='C:/Users/DELL/Documents/datasets/jewellery_abdallah_dataset/dataset/Earing/earings13.jpg'
img = image.load_img(img_path,target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)  
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='cosine')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

print(distances)

for file in indices[0][0:5]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)
    
    
    
#910 images of earrings total + 162 images of necklace + 116 images of rings

#shape,diamond and shining params for checking result and comparing for result
#data augmentation of images that are less
#references at end of report

