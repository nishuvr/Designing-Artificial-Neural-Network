#Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

epoch=100# number of training iterations
lr=0.1 # learning 
inputlayer_neurons = 9# number of features in data set
hiddenlayer_neurons= 5# number of hidden layers neurons
output_neurons = 1
def sigmoid(x):
    return 1/(1+np.exp(-x))
	
def sig_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_prime(z):
	return 1 - np.power(tanh(z), 2)
	

class NN:

	''' X and Y are dataframes '''
	#Neural Network Model
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
		#Defining the neural network structure
		global w1
		global w2
		global w3
		global bias1
		global bias2
		global bias3
		#Initializing the model's parameters
		''' 
  			The input units are equal to the number of features in the dataset,
  			hidden layers are set to 5 (for this purpose), 
     		and the problem is the binary classification we will use a single layer output 
        '''
    	#Weight and biases are initialized randomly
		w1=np.random.rand(inputlayer_neurons,hiddenlayer_neurons)
		w2=np.random.rand(hiddenlayer_neurons,hiddenlayer_neurons)
		w3=np.random.rand(hiddenlayer_neurons,output_neurons)
		bias1=np.random.rand(1)
		bias2=np.random.rand(1)
		bias3=np.random.rand(1)
		#learning_rate
		lr=0.5
		'''
  		Loop:
    		- Implementing forward propagation
    		- Implementing backward propagation to get the gradients
    		- Updating parameters 
      	'''
		for ite in range(0,epoch):
			#forward propagation
			'''
			For forward propagation, given the set of input features (X), 
   			we need to compute the activation function for each layer. 
      		For the 2 hidden layers, we are using sigmoid activation functions.
			Similarly, for the output layer, we are using tanh activation function.
			'''
			w1_out=np.dot(X,w1)+bias1
			out1=sigmoid(w1_out)
			w2_out=np.dot(out1,w2)+bias2
			out2=sigmoid(w2_out)
			w3_out=np.dot(out2,w3)+bias3
			out3=tanh(w3_out)
			#back propagation - We need to calculate the gradient with respect to different parameters using backward propagtaion
			error=np.square(out3-Y)/2
			slope_output_layer=sig_der(out3)
			d_output1=error*slope_output_layer
			d_output2=sig_der(out2)*np.dot(d_output1,w3.T)
			d_output3=tanh_prime(out1)*np.dot(d_output2,w2.T)
			#updating the parameters
			w3+=(np.dot(out2.T*lr,d_output1))
			w2+=(np.dot(out1.T*lr,d_output2))
			w1+=(np.dot(X.T*lr,d_output3))
			for i in d_output1:
				bias3=bias3+lr*i
			for i in d_output2:
				bias2=bias2+lr*i
			for i in d_output3:
				bias1=bias1+lr*i
		return out3

    #Using the learned parameter, we can predict the output for each example   
	def predict(self,X):
		z1=np.dot(X,w1)+bias1
		a1=sigmoid(z1)
		z2=np.dot(a1,w2)+bias2
		a2=sigmoid(z2)
		z3=np.dot(a2,w3)+bias3
		a3=sigmoid(z3)
		prediction=a3
		return prediction

	def CM(self,y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		accuracy=(tp+tn)/(tp+fp+tn+fn)
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")
		print(f"Accuracy : {accuracy}")

if __name__=="__main__":
    global df
    #dataset
    df=pd.read_csv("clean.csv")
    df=df.iloc[:,1:]
    global X
    global Y
    X=df.loc[:,df.columns != 'Result']
    Y=df.iloc[:,-1]	
    X=X.to_numpy()
    Y=Y.to_numpy()
    model=NN()
    #dividing the dataset into a training set(70%) and test set(30%) 
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.70,test_size=0.30,random_state=0)
    Y_train=Y_train.reshape(len(Y_train),1)
    s=model.fit(X_train,Y_train)
    p=model.predict(X_test)
    Y_test=Y_test.reshape(len(Y_test),1)
    model.CM(Y_test,p)