# Part-of-Speach tagger
* preprocessing.py 
1. converts the feature dictionary to numpy array
* test.py 
1. load data created by preprocessing.py
2. import the model from model.py
3. fit the model using training data and validation data
4. evaluate the model and print out the score


## Result
```
Train
macro =  0.9872662071305855
micro =  0.98685771309145
weighted =  0.986850374587165
Test
macro =  0.8827882981417612
micro =  0.9260348243144615
weighted =  0.9259446423017527
```