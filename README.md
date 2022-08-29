# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Deep learning is the development of deep learning algorithms that can be used to train and predict output from complex data.The word “deep” in Deep Learning refers to the number of hidden layers i.e. depth of the neural network. Essentially, every neural network with more than three layers, that is, including the Input Layer and Output Layer can be considered a Deep Learning Model.TensorFlow, an open-source software library for machine learning, offers a robust framework for implementing neural network regression models.The Reluactivation function helps neural networks form deep learning models. Due to the vanishing gradient issues in different layers, you cannot use the hyperbolic tangent and sigmoid activation. You can overcome the gradient problems through the Relu activation function.


## Neural Network Model

![dl](https://user-images.githubusercontent.com/75235818/187118568-d080b589-13d2-4091-8307-f93a1c69bc38.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model with hidden layer 1-3, hidden layer 2-6 and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data=pd.read_csv("dataset1.csv")
data.head()
x=data[['Input']].values
x
y=data[['Output']].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
Scaler.fit(x_test)
x_train1=Scaler.transform(x_train)
x_train1
x_train
AI_BRAIN=Sequential([
    Dense(3,activation='relu'),
    Dense(6,activation='relu'),
    Dense(1)
])
AI_BRAIN.compile(optimizer='rmsprop', loss='mse')
AI_BRAIN.fit(x_train1,y_train,epochs=2000)
loss_df=pd.DataFrame(AI_BRAIN.history.history)
loss_df.plot()
x_test1=Scaler.transform(x_test)
x_test1
AI_BRAIN.evaluate(x_test1,y_test)
x_n1=[[25]]
x_n1_1=Scaler.transform(x_n1)
AI_BRAIN.predict(x_n1_1)

```

## Dataset Information

![data](https://user-images.githubusercontent.com/75235818/187115162-0f3b2379-80cf-4bec-851c-a27c6f0fe901.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![1EE5F5EF2ACD4D6EB8300CE1DDD6327E](https://user-images.githubusercontent.com/75235818/187119465-5dd19d12-2b58-4d73-9c7e-bb2c318a2984.png)

### Test Data Root Mean Squared Error

![6424F3429EAD4DF8A15D0017826F4B28](https://user-images.githubusercontent.com/75235818/187119491-9aec6ea6-6f83-402a-a746-4436f08a67ea.png)

### New Sample Data Prediction

![FEDB1DA7BC654A56B2724A12ACBCCBBF](https://user-images.githubusercontent.com/75235818/187119543-315733ae-0a59-41fb-86f8-237713de5425.png)

## RESULT

Thus,the neural network regression model for the given dataset is developed.
