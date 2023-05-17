import numpy as np
# This library is used for Mathematical operations in python
import cv2
#This library(open-cv) is used to for inacting Computer Vision (Here, image processing purposes)
import os
#for creating a path outline to save the colorized image.



print("Executing models.....")
#Prints: "Executing model"
name = input("Please input the name of your new image:")

net = cv2.dnn.readNetFromCaffe('Deploy.prototxt','Colorize.caffemodel')
#This syntax above lets the program to call out two model using open cv deep learning

pts = np.load('Hull.npy')
#load() function return the input array from a disk file with npy extension(.npy).


#Creating a Data Access Layer from Open-cv
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2,313,1,1)
#The above command allows for the transposing of data and reshape it according to our use

#Another data access layer that pulls the
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]


#The input to the network is trained and the network is used to predict the outcomes.
#Load the input image from imread function present in OpenCV, Scale the image accordingly.
image = cv2.imread('6.png')
scaled = image.astype("float32")/255.0
#After loading the images, convert all images from the one color space to other color spaces respectively.
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)


#We will resize the input image to 224×224, the required input dimensions for the neural network;
#Scaling the predicted volume to the same coordinates as our input image;
#After this, we scale the predicted volume to be the same coordinates as our input image;
#We are also reshaping the image;
#The channel from the original image is appended with the predicted channels:
resized = cv2.resize(lab,(224,224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))
ab = cv2.resize(ab, (image.shape[1],image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

#Finally, we Convert the Greyscale image from one random color space to the standard space of color
# And hence, obtain the colorized image of our original grayscale image.
colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)
colorized = (255 * colorized).astype("uint8")

#Displaying the output result:
cv2.imshow("Greyscale:",image)
cv2.imshow("RGB Image:",colorized)
cv2.waitKey(0)
path = 'C:/Users/terra/PycharmProjects/colorized pictures'
cv2.imwrite(os.path.join(path , name+'_color.jpg'), colorized)

#path = os.path.join(, name + '_edited.jpg')

#cv2.imwrite(os.path.join('C:\Users \ terra\PycharmProjects \ colorized pictures''lion.jpg'),colorized)