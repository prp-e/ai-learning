from numpy import exp, dot, random, array

class neuralNetwork():
 def __init__(self):
  random.seed(1)
  self.synaptic_weights = 2 * random.random((3, 1)) - 1

 def __sigmoid(self, x):
  return 1 / (1 + exp(-x))

 def __sigmoid_deriv(self, x):
  return x * (1 - x)

 def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
  for iteration in xrange(number_of_iterations): 
   output = self.think(training_set_inputs)
   error = training_set_outputs - output
   adjustment = dot(training_set_inputs.T, error * self.__sigmoid_deriv(output))
   self.synaptic_weights += adjustment

 def think(self, inputs):
  return self.__sigmoid(dot(inputs, self.synaptic_weights)) 


if __name__ == "__main__" :
	
	neural_net = neuralNetwork()

	print "Random synaptic weights: "
	print neural_net.synaptic_weights
	
	training_inputs = array([[1, 1, 0], [1, 1, 1], [0,1,0], [0,1,1]])
	training_outputs = array([[1, 0, 0, 1]]).T

	neural_net.train(training_inputs, training_outputs, 10000) 

	print "New weights: "
 	print neural_net.synaptic_weights 

	print "New situation like [1,0,1]?"
	print neural_net.think(array([1,0,1]))
