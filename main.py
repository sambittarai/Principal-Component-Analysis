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

#Project into PCA subspace
X_transformed = np.dot(X_train, components.T)
print(X_transformed.shape)
X_test_transformed = np.dot(X_test, components.T)
print(X_test_transformed.shape)

#Finally, plot the resulting eigen-vectors of the face PCA model, AKA the eigenffaces
import matplotlib.pyplot as plt  

#Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
	"""Helper function to plot a gallery of portraits"""
	plt.figure(figsize=(1.8*n_col, 2.4*n_row))
	plt.subplots_adjust(bottom=0, left=.01, right=.99, top=90, hspace=.35)

	for i in range(n_row*n_col):
		plt.subplot(n_row, n_col, i+1)
		plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
		plt.title(titles[i], size=12)
		plt.xticks(())
		plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
  
plt.show()

#We should always evaluate the performance of the dimensionality reduction via a compactness plot
explained_variance = (S**2)/(n_samples - 1)
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance/total_var
ratio_cumsum = np.cumsum(explained_variance_ratio)
print(ratio_cumsum.shape)
eigenvalueCount = np.arange(n_components)

plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()

#By taking the top 150 dimensions of data with highest variance we can represent 95% of our data

#Use the PCA 'face space' as features and build a random forest classifier to classify the faces according to the labels.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#Build the random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train) # expects X as [n_samples, n_features]

predictions = estimator.predict(X_test_transformed)
correct = predictions==y_test
total_test = len(X_test_transformed)

#Results
print("Total Testing:", total_test)
#print("Predictions:", predictions)
#print("Which Correct:", correct)
print("Total Correct:", np.sum(correct))
print("Accuracy:", np.sum(correct)/total_test)

print(classification_report(y_test, predictions, target_names=target_names))
