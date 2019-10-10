#1
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np 

#Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape 

print("n_samples :", n_samples)
print("Height :", h)
print("Width :", w)

lfw_people.images.shape #Shape of the data

#for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
#

X = lfw_people.data 
#n_features = X.shape[1]
#X.shape - (1288, 1850) -> shows the shape of the data
#X.shape[1] - 1850 -> shows the shape of the data in the 2nd dimension
#X.shape[0] - 1288

n_features = X.shape[1] #Number of features, which corresponds to all the pixels of the image

#The label to predict is the id of the person

#There are a total of 7 labels/classes corresponding to the images at hand, 
#The names of the classes is stored in "target_names"
y = lfw_people.target 
target_names = lfw_people.target_names # ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush', 'Gerhard Schroeder', 'Hugo Chavez', 'Tony Blair']
n_classes = target_names.shape[0] # shape is (7,), because we have 7 labels

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

#Split into a training and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#X_train.shape -> (966, 1850) -> No. of training data
#y_train.shape -> (966,) -> No. of training labels
#Similarly for testing set

#Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled dataset)
#Unsupervised feature extraction/dimensionality reduction

n_components = 150 #Top 150 dimensions with highest variance

#Center data
mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean

#Eigen-decomposition
U, S, V = np.linalg.svd(X_train, full_matrices=False)
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

#eigenfaces.shape -> (150, 50, 37)
#components.shape -> (150, 1850)

#U.shape -> (966, 966)
#V.shape -> (966, 1850)