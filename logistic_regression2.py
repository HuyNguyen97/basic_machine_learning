import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt



DATA_SET_PATH_x = "ex4x.txt"
DATA_SET_PATH_y = "ex4y.txt"
data = pd.read_csv(DATA_SET_PATH_x)            
data = pd.read_csv(DATA_SET_PATH_y)    
                           
train_x = np.loadtxt('ex4x.txt')                                         # convert file(two column and 80 row) to array to train
#print(train_x)  
train_y = np.loadtxt('ex4y.txt')                                         # convert file(two column and 80 row) to array to train
#print(train_y)  
                         
train_x = np.concatenate((np.ones((1,80)),train_x.T), axis = 0)          #add one column value 1 to train_x( a trick for theta_0)
train_x=train_x.T
#print(train_x[2,1])
def logistic_function(x):
	return 1/(1+np.exp(-x))

def gradient(theta_init, X, Y, iteration, learning_rate ):
	theta = [theta_init]
	count = 0
	N = X.shape[0]
	while count < iteration:                                              #number of iteration
	       #mix_data=train_x[np.random.permutation(train_x.shape[0]),:] 
	       #shuffle data(between x dimension)
	    np.random.seed(2)
	    mix_id = np.random.permutation(N)
	    for i in mix_id:
	    	xi = X[i,:].reshape(1,3)                                      #extract row 1 in train_x list
	    	xi=xi.T
	    	yi = Y[i]
	    	h = logistic_function(np.dot(theta[-1],xi)) 
	    	#print (xi)
	    	#print(yi)
	    	#print(theta[-1])    
	    	
	    	theta_new = theta[-1] - learning_rate*(h-yi)*xi.T              #update theta parameter
	    	theta.append(theta_new)     
	    	              
	    	count += 1
	return theta

#def logistic_loss_function()

	

theta_init = np.random.randn(1, 3)


theta = gradient(theta_init, train_x, train_y,50000, 0.005)
print(theta[-1])

numtrain = range(0,80)
right=0   # test on train set
wrong=0

for i in numtrain:
	xi = train_x[i,:].reshape(1,3)                                      
	xi=xi.T
	yi=train_y[i]
	h = logistic_function(np.dot(theta[-1],xi)) 
	print(h)
	if(h>=0.5 and yi==1) or (h<0.5 and yi==0):
		right+=1
	else:
		wrong+=1

print(right)
print(wrong)


