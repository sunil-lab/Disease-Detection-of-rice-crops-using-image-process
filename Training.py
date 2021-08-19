import cv2
import matplotlib.pyplot as mt
import glob
import sys
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
hog = cv2.HOGDescriptor()
features = []
labels=[]
for img in glob.glob("E:/LIDS/green leaves/*.jpg"):
    image= cv2.imread(img)
    image=cv2.resize(image,(150, 150))
    h = hog.compute(image)
features.append(h)
labels.append(0)
print('normal ', len(features))
for img in glob.glob("E:/LIDS/Brown spot/*.jpg"):
    image= cv2.imread(img)
    image=cv2.resize(image,(150, 150))
    h = hog.compute(image)
features.append(h)
labels.append(1)
print('brown ',len(features))
for img in glob.glob("E:/LIDS/paddy blast/*.jpg"):
    image= cv2.imread(img)
    image=cv2.resize(image,(150, 150))
    h = hog.compute(image)
features.append(h)
labels.append(2)
print('paddyblast ', len(features))
for img in glob.glob("E:/LIDS/bacterial blight/*.jpg"):
    image= cv2.imread(img)
    image=cv2.resize(image,(150, 150))
    h = hog.compute(image)
features.append(h)
labels.append(3)
print('bacterial ', len(features))

fet= np.array( features )
features=[]
lb=np.array(labels)
lb=np.reshape(lb,[30, 1])
fet=np.reshape(fet,[30,124740])
print(np.shape(fet))
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(fet,lb)
pickle.dump(clf,open('neural2.model', 'wb'))
