import tflearn 

# data 

x = [1, 3, 4, 5, 3, 2, 6, 7, 4, 4.5, 1.2, 8.5, 3, 1.24, 3.3, 6.55] 
y = [1000, 5600, 2300, 5000, 3200, 4500, 4000, 3200, 2200, 1100, 8000, 7240, 1200, 4100, 1200, 6500]

input_ = tflearn.input_data(shape=[None]) 
linear = tflearn.single_unit(input_) 
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01) 

m = tflearn.DNN(regression) 
m.fit(x, y, n_epoch=1000, show_metric=True, snapshot_epoch=False) 

print "\nRegression Result: " 

print "Y = " + str(m.get_weights(linear.W)) 
print "X = " + str(m.get_weights(linear.b))

print "\n Test prediction for x = 3.5, 2.5, 4.5" 
print m.predict([3.5, 2.5, 4.5])
