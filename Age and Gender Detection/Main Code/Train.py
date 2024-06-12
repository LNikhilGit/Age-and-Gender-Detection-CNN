import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout,Dense,Flatten,MaxPooling2D,Conv2D,Input,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

epochs=10
totalimages=100


fldr="./UTKFace"
files=os.listdir(fldr)
ages=[]
genders=[]
images=[]
n=totalimages
for fle in files:
  n-=1
  if(n==0):
    break
  age=int(fle.split('_')[0])
  gender=int(fle.split('_')[1])
  total=fldr+'/'+fle
  print(total)
  image=cv2.imread(total)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image= cv2.resize(image,(48,48))
  images.append(image)

n = totalimages
for fle in files:
  n -= 1
  if (n == 0):
    break
  age=int(fle.split('_')[0])
  gender=int(fle.split('_')[1])
  ages.append(age)
  genders.append(gender)


images_f=np.array(images)

labels=[]

i=0
while i<len(ages):
  label=[]
  label.append([ages[i]])
  label.append([genders[i]])
  labels.append(label)
  i+=1

images_f_2=images_f/255
labels_f=np.array(labels)
X_train, X_test, Y_train, Y_test= train_test_split(images_f_2, labels_f,test_size=0.25)
Y_train_2=[Y_train[:,1],Y_train[:,0]]
Y_test_2=[Y_test[:,1],Y_test[:,0]]


def Convolution(input_tensor, filters):
  x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(
    input_tensor)
  x = Dropout(0.1)(x)
  x = Activation('relu')(x)

  return x


def model(input_shape):
  inputs = Input((input_shape))

  conv_1 = Convolution(inputs, 32)
  maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
  conv_2 = Convolution(maxp_1, 64)
  maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
  conv_3 = Convolution(maxp_2, 128)
  maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
  conv_4 = Convolution(maxp_3, 256)
  maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
  flatten = Flatten()(maxp_4)
  dense_1 = Dense(64, activation='relu')(flatten)
  dense_2 = Dense(64, activation='relu')(flatten)
  drop_1 = Dropout(0.2)(dense_1)
  drop_2 = Dropout(0.2)(dense_2)
  output_1 = Dense(1, activation="sigmoid", name='sex_out')(drop_1)
  output_2 = Dense(1, activation="relu", name='age_out')(drop_2)
  model = Model(inputs=[inputs], outputs=[output_1, output_2])
  model.compile(loss=["binary_crossentropy", "mae"], optimizer="Adam",
                metrics=["accuracy"])

  return model

Model=model((48,48,3))

History=Model.fit(X_train,Y_train_2,batch_size=64,validation_data=(X_test,Y_test_2),epochs=epochs)
Model.evaluate(X_test,Y_test_2)
Model.save("Model.h5")
pred=Model.predict(X_test)
i=0
Pred_l=[]
while(i<len(pred[0])):

  Pred_l.append(int(np.round(pred[0][i])))
  i+=1

report=classification_report(Y_test_2[0], Pred_l)
print(report)