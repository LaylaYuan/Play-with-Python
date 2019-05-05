import sys
import optparse

#create a new option parser
parser = optparse.OptionParser()
#add an option to look for the -f 
parser.add_option('-f', '--file', dest='fileName', help='file name to read from')

#get the options entered by the user at the terminal
(options, others) = parser.parse_args()

usingFile = False

#inspect the options entered by the user!
if options.fileName is None:
	print "DEBUG: the user did not enter the -f option"
else:
	print "DEBUG: the user entered the -f option"
	usingFile = True

X = []
if(usingFile == True):
	#attempt to open and read out of the file
	print "DEBUG: the file name entered was: ", options.fileName
	file = open(options.fileName, "r") 
	for line in file:
		line = line.strip().split(',')
		newline = [float(i) for i in line] #convert string in each line into number(floating type)
		X.append(newline)		

from sklearn import cluster
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import random

#normalize the data 
def normalization(X,max_X,min_X):
    X = (X-min_X)/(max_X - min_X)
    return X

X = normalization (X,np.max(X),np.min(X))
X = np.array(X)
def _distance(p1,p2):
		Distance = np.sum((p1-p2)**2)
		return np.square(Distance)

Z = linkage(X,method='ward',metric='euclidean')
dendrogram(Z,leaf_rotation=30, leaf_font_size=2)

plt.title('hierarchical clustering dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.axhline(y = 3, color='black')
plt.show()
print 'After Normalization,we chose distance = 3 as threshold, where k = 3'
centroids = []#randomly initialize three threshold
i = 0
k = 3
while i < k:
	centroids.append(X[random.randint(0,X.shape[0])])
	i += 1

#print centroids

def _converged(old_centriods, centroids):
		# if centroids not changed, we say 'converged'
		set1 = set([tuple(c) for c in old_centriods])
		set2 = set([tuple(c) for c in centroids])
		return (set1 == set2)

converged = False

j = 0
while converged == False:
#calculate the distance between each point and threshold
	old_centriods = np.copy(centroids)
	Cluster1 = []
	Cluster2 = []
	Cluster3 = []
	for i in range(X.shape[0]):
		Distance1 = _distance(X[i],centroids[0])
		Distance2 = _distance(X[i],centroids[1])
		Distance3 = _distance(X[i],centroids[2])
		if min(Distance1,Distance2,Distance3) == Distance1:
			Cluster1.append(X[i])
		elif min(Distance1,Distance2,Distance3) == Distance2:
			Cluster2.append(X[i])
		elif min(Distance1,Distance2,Distance3) == Distance3:
			Cluster3.append(X[i])

	Cluster1 = np.array(Cluster1)
	Cluster2 = np.array(Cluster2)
	Cluster3 = np.array(Cluster3)
	centroids = []
	#recompute the center centroids
	centroids.append([np.mean(Cluster1[:, 0]), np.mean(Cluster1[:, 1])])
	centroids.append([np.mean(Cluster2[:, 0]), np.mean(Cluster2[:, 1])])
	centroids.append([np.mean(Cluster3[:, 0]), np.mean(Cluster3[:, 1])])

    
	#check whether the centriods will change or not
	converged = _converged(old_centriods,centroids)
	print 'Now after',j,'times iteraton, the converged is',converged
	
	if j == 0 or j == 5 or j == 10 or j == 100 or converged:
		plt.plot(Cluster1[:,0],Cluster1[:,1],'xr')
		plt.plot(Cluster2[:,0],Cluster2[:,1],'xb')
		plt.plot(Cluster3[:,0],Cluster3[:,1],'xg')
		plt.scatter(centroids[0][0],centroids[0][1],c = 'black')
		plt.scatter(centroids[1][0],centroids[1][1],c = 'black')
		plt.scatter(centroids[2][0],centroids[2][1],c = 'black')
		plt.title('Clustering after iteration %s' % j)
		plt.show()
	j += 1