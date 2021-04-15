
#Designing Artificial Neural Networks for classification of LBW Cases from Scratch

->Implementation:
-Data pre-processing:
	.Rows containing null values in either ‘Age’, ‘Residence’, ‘Education’ columns have been dropped.
	.Missing values in ‘Weight’ column were replaced by the mean weight of columns with similar age.
	.Missing values in ‘BP’ column were replaced by mean BP of columns with same age and weights close to the weight in our target row.
-Neural Network:
	sigmoid(x): It’s an activation function, returns the sigmoid value of the input
	sig_der(x): returns the derivative of sigmoid of the input
	tan_h(x): It’s an activation function, returns the tanh of the input
	tanh_prime(x): returns the derivative of tanh of the input.
	. All weights and biases have been randomly initialised and are updated iteratively over epoch number of times.
	. The hidden layers use ‘sigmoid’ as activation function whereas the output layer uses ‘tanh’ as the activation function.

->Hyper parameters involved:
The model represents an Artificial Neural Network with the following parameters
	- Layers: Input layer, 2 Hidden layers, Output layer
	-Number of neurons per layer: Hidden layers->5 neurons each
				      Output layer  ->1 neuron

->The model considers all the inputs,weights as vectors so as to make use of vector properties like dot product to reduce complexity.

->Execution:
	Python Neural_Net.py	
			
