import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.models import Sequential
from keras.optimizers import Adam


#Change access path of features and target as relevent to our project. You may need to rethink the loop logic
features=[]
target=[]
for x in range(0,3):
  ImagesNamesList=os.listdir("AgeData" + "/" + str(x))
  for y in ImagesNamesList:
    Imgarr=cv2.imread("AgeData" + "/" + str(x) + "/" + y)
    try:
      Imgarr = cv2.resize(Imgarr, (100, 100))  # Here 100,100 implies dimension
    except:  # this is used for exception
      pass
    features.append(Imgarr)
    target.append(x)
features=np.array(features)
target=np.array(target)
print(features.shape,target.shape)
features_train,features_test,target_train,target_test=train_test_split(features, target,test_size=0.2)
print(features_train.shape,target_train.shape)

def preprocessing(image):
  image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  image=image/255
  return image

features_train=np.array(list(map(preprocessing,features_train)))
print(features_train.shape)
features_train=features_train.reshape(11120,100,100,1)
features_test=np.array(list(map(preprocessing,features_test)))
print(features_test.shape)
features_test=features_test.reshape(2781,100,100,1)
dataGen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1)
dataGen.fit(features_train)
batches=dataGen.flow(features_train,target_train,batch_size=20)
print(len(batches))
images,labels=next(batches)
print(images.shape)
plt.imshow(images[0].reshape(100,100))
plt.show()
plt.figure(figsize=(10,10))
for i in range(0,20):
    plt.subplot(4,5,i+1)
    plt.imshow(images[i].reshape(100,100))
plt.show()
target_train=to_categorical(target_train)
print(target_train.shape,features_train.shape)

model=Sequential()
model.add(Conv2D(60,(3,3),activation="relu",input_shape=(100,100,1)))
model.add(Conv2D(60,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(30,(3,3),activation="relu"))
model.add(Conv2D(30,(3,3),activation="relu"))
model.add(Conv2D(30,(3,3),activation="relu"))
model.add(MaxPooling2D((2,2)))
#Dropout is used to block some neurons on overfitting
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.compile(Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.fit_generator(dataGen.flow(features_train,target_train,batch_size=20),epochs=20)
model.predict(features_test)


#converting model to json file to save
model_json=model.to_json()
with open("age.json", "w") as abc:
  abc.write(model_json)
  abc.close
#We save the weights of the model too in h5 format
model.save_weights("ageweights.h5")
print("Saved the model")

