from sklearn import datasets
import matplotlib.pyplot as plt
import random

# before define the model, define b and x as list
b = []
for i in range(14):
    a =0
    b.append(a)
x = []
def model(b,x):
    p = 0
    i = 0
    while i in range(0,len(x)):
        p = p + x[i]*b[i]
        i += 1
    return p

def RMSE(predictions,y,n):
	s = 0
	for i in range(0,n):
		s = s + (predictions[i]-y[i])**2
		r = (s/n)**0.5
	return r

boston = datasets.load_boston()
data = boston.data.tolist()
y = boston.target
feature = boston.feature_names.tolist()

max_epoch = 10
epoch = []
i = 0
while i in range(max_epoch):
    epoch.append(i)
    i += 1

trainingX = []
trainingY = []
testX = []
testY = []


index = 0
learning_rate = [0.00001,0.0001,0.001,0.01,0.1,1 ]

table = []
feature.append('TARGET') 
min_feature = []
max_feature = []
for i in range(0,len(data)):
    data[i].append(y[i])
    row = data[i]
    table.append(row)

#normalize data
for n in range(len(feature)):
    min_feature.append(table[0][n])
    max_feature.append(table[0][n])
    for row in table:
        if row[n] < min_feature[n]:
            min_feature[n] = row[n]
        if row[n] > max_feature[n]:
            max_feature[n] = row[n]

for row in table:
    for i in range(len(feature)):
        row[i] = (row[i]-min_feature[i])/(max_feature[i]- min_feature[i])
for i in range(len(data)):
    data[i].insert(0,1)
    data[i].pop()
    if (i%10 == 0):
        testX.append(data[i])
        testY.append(y[i])
    else:
        trainingX.append(data[i])
        trainingY.append(y[i])

for index in range(len(learning_rate)):
    current_epoch = 0 
    R = []       
    while current_epoch < max_epoch:
        for i in range(len(trainingX)):
            x = trainingX[i]
            error =(model(b,x)-trainingY[i])/len(trainingX) 
            for n in range(14):
                b[n] = b[n]-learning_rate[index] * error *x[n]
            
        predictions = []
        for i in range(0,len(testX)):
	        prediction = model(b,testX[i])
	        predictions.append(prediction)

        RMSE(predictions,testY,len(testY))
        R.append(RMSE(predictions,testY,len(testY)))
        print "Now the learning rate is:",learning_rate[index],"After epoch",current_epoch,"the RMSE we found is:",RMSE(predictions,testY,len(testY))
        #random.shuffle(trainingX)
        current_epoch += 1
    dummy = raw_input("press the <ENTER> key to show the PLOT")
    print '*****************************************************************' #for easy reading
    
    plt.scatter(epoch,R,c = 'blue')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('learning_rate = {n}'.format(n = learning_rate[index]),loc = 'center')
    plt.plot(epoch,R)
    plt.show()

print 'PROGRAM COMPLETE'    