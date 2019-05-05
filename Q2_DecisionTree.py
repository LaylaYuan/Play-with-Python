

import sys
import optparse
import numpy as np
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

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

data = []
target = []
trainingX1 = []
trainingY1 = []
testX1 = []
testY1 = [] 
if(usingFile == True):
	#attempt to open and read out of the file
	print "DEBUG: the file name entered was: ", options.fileName
	file = open(options.fileName, "r") 
	for line in file:
		line = line.strip()
		fields = line.split(',')			
		if fields[len(fields)-1] == 'Very Late':
			Adpt_Class = 1
		else:
			Adpt_Class = 0
		target.append(Adpt_Class)
		data.append(fields)

newData = []
for i in range(len(data)):
	newRow = data[i][1:9]
	newData.append(newRow)


#transfer categorical data into numerical data 
df = pd.DataFrame(data = newData)
df.columns = ['Gender','Age','Marital_Status','Current_Plan',
'Payment_Method','Contract_Length','Has_Kids','Other_Services_Bundled']


G_mapping = {'F':0,'M':1}
MS_mapping = {'Married':0, 'Single': 1}
CP_mapping = {'PrePaid' : 0, 'Low' : 1, 'Medium': 2, 'Heavy': 3}
PM_mapping = {'Automatic': 0,'Non-Automatic': 1}
CL_mapping = {'No Contract': 0, '12 Months':1, '24 months':2,'36 Months':3}
HK_mapping = {'N':0, 'Y':1}
Other_mapping = {'N':0, 'Y':1}

df['Gender']=df['Gender'].map(G_mapping)
df['Marital_Status']=df['Marital_Status'].map(MS_mapping)
df['Current_Plan']=df['Current_Plan'].map(CP_mapping)
df['Payment_Method']=df['Payment_Method'].map(PM_mapping)
df['Contract_Length']=df['Contract_Length'].map(CL_mapping)
df['Has_Kids']= df['Has_Kids'].map(HK_mapping)
df['Other_Services_Bundled'] = df['Other_Services_Bundled'].map(Other_mapping)

#print df
#print NewArray
#print len(NewArray)

NewArray=np.array(df) #convert dataframe into array
for i in range(len(NewArray)):
	if (i % 10 == 0):
		testX1.append(NewArray[i])
		testY1.append(target[i])
	else:
		trainingX1.append(NewArray[i])
		trainingY1.append(target[i])


depth = [None,2,4,8,16]
nodes = [2,4,8,16,32,64,128,256]
for n in range(len(depth)):
	accurate = []
	print '      ' #for easy reading 
	print 'current max_depth is:', depth[n]
	for i in range(len(nodes)):
		clf = tree.DecisionTreeClassifier(max_leaf_nodes =nodes[i],max_depth=depth[n] )
		clf.fit(trainingX1,trainingY1)
		predictions = clf.predict(testX1)
		print '-------------------------------------------------' #for easy reading 
		print 'current max_leaf_nodes is:', nodes[i]
		#print 'model predictions are:',predictions
		correct = 0
		incorrect = 0
		for i in range(len(predictions)):
			if predictions[i] == testY1[i]:
				correct += 1
			else:
				incorrect += 1
		print 'correct predictions are:',correct,'incorrect predictions are:', incorrect

		accuracy = float(correct)/(incorrect+correct)
		accurate.append(accuracy)
		print 'model accuracy :', accuracy
		
	print '        '#for easy reading
	print 'LIST the Max_Leaf_Nodes:',nodes
	print 'LIST the Model Accuracy under max_depth =',depth[n],":",accurate
	dummy = raw_input("press the <ENTER> key to show the plot")

	plt.scatter(nodes,accurate)
	plt.plot(nodes,accurate)
	plt.xlabel('Max_Leaf_Nodes')
	plt.ylabel('Accuracy')
	plt.title('Max_Depth = {n}'.format(n = depth[n]))
	plt.show()
    

print 'now pruning the data into 50:50'
dummy = raw_input("press the <ENTER> key to continue")

VeryLate = []
NLate = []
LateFeature= []
NLFeature = []
#split data into two groups: 'Very Late' and 'Not Very Late'
for i in range(len(target)):
	if target[i] == 1:
		LateFeature.append(NewArray[i])
		VeryLate.append(target[i])
	else:
		NLFeature.append(NewArray[i])
		NLate.append(target[i])

#prune 'Not Very Late' data to avoid bias
Prune_NLate = []
Prune_NLfeature = []
Prune_NLfeature = NLFeature[:len(LateFeature)]
Prune_NLate = NLate[:len(VeryLate)]
#print len(Prune_NLate), len(Prune_NLfeature)


trainingX2 = []
trainingY2 = []
testX2 = []
testY2 = [] 
for i in range(len(VeryLate)):
	if i % 10 == 0:
		testX2.append(LateFeature[i])
		testY2.append(VeryLate[i])
	else:
		trainingX2.append(LateFeature[i])
		trainingY2.append(VeryLate[i])

for i in range(len(Prune_NLate)):
	if i % 10 == 0:
		testX2.append(Prune_NLfeature[i])
		testY2.append(Prune_NLate[i])
	else:
		trainingX2.append(Prune_NLfeature[i])
		trainingY2.append(Prune_NLate[i])

for n in range(len(depth)):
	accurate = []
	print '                  ' #for easy reading
	print 'current max_depth is:', depth[n]
	for i in range(len(nodes)):
		clf = tree.DecisionTreeClassifier(max_leaf_nodes =nodes[i],max_depth=depth[n] )
		clf.fit(trainingX2,trainingY2)
		predictions = clf.predict(testX2)
		print '-----------------------------------------------------------------' #for easy reading 
		print 'current max_leaf_nodes is:', nodes[i]
		#print 'model predictions are:',predictions
		correct = 0
		incorrect = 0
		for i in range(len(predictions)):
			if predictions[i] == testY2[i]:
				correct += 1
			else:
				incorrect += 1
		print 'correct predictions are:',correct,'incorrect predictions are:', incorrect

		accuracy = float(correct)/(incorrect+correct)
		accurate.append(accuracy)
		print 'Model Accuracy :', accuracy

	print '        '#for easy reading
	print 'LIST the Max_Leaf_Nodes:',nodes
	print 'LIST the Model Accuracy under max_depth =',depth[n],":",accurate
	dummy = raw_input("press the <ENTER> key to show the plot")

	plt.scatter(nodes,accurate)
	plt.plot(nodes,accurate)
	plt.xlabel('Max_Leaf_Nodes')
	plt.ylabel('Accuracy')
	plt.title('Max_Depth = {n}'.format(n = depth[n]))
	plt.show()

print 'PROGRAM COMPLETE'