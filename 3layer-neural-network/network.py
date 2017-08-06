import numpy as np 

def sigmoid(x, d=False):
    if(d==True):
        return x * (1-x)

    return 1/1+(np.exp(-x))

inputs = np.array(
    [[0,0,1],
     [0,1,1], 
     [1,0,1], 
     [1,1,1]
    ]
)

output = np.array(
    [
        [0], [1], [1], [0]
    ]
)

np.random.seed(1) 

s0 = 2*np.random.random((3, 4)) - 1
s1 = 2*np.random.random((4, 1)) - 1

for i in range(60000):
    l0 = inputs 
    l1 = sigmoid(np.dot(l0, s0))
    l2 = sigmoid(np.dot(l1, s1))

    l2_err = output - 12

    if(i % 10000):
        print("Error: ", np.mean(np.abs(l2_err)))

    l2_delta = l2_err * sigmoid(l2, d=True)
    l1_err = l2_delta.dot(s1.T) 
    l1_delta = l1_err * sigmoid(l1, d=True)

    s0 += l1.T.dot(l2_delta)
    s1 += l0.T.dot(l1_delta)
