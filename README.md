# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

*A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Deep learning is the development of deep learning algorithms that can be used to train and predict output from complex data.The word “deep” in Deep Learning refers to the number of hidden layers i.e. depth of the neural network. Essentially, every neural network with more than three layers, that is, including the Input Layer and Output Layer can be considered a Deep Learning Model.TensorFlow, an open-source software library for machine learning, offers a robust framework for implementing neural network regression models.The Reluactivation function helps neural networks form deep learning models. Due to the vanishing gradient issues in different layers, you cannot use the hyperbolic tangent and sigmoid activation. You can overcome the gradient problems through the Relu activation function.


## Neural Network Model

![image](https://user-images.githubusercontent.com/75235293/187087815-a10dcad4-4746-47dd-951b-84f60e245d6c.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

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
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1,activation='relu')
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

![image](https://user-images.githubusercontent.com/75235293/187087226-b6596c46-ed43-4e53-9abd-3776b0ce9d8f.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235293/187086647-6e61584c-5f94-4d52-a31c-d4189daa984b.png)


### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75235293/187086691-3307bc1e-5eac-4718-8e94-89d7729125cd.png)


### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/75235293/187086786-cf327d8c-756e-4df5-8528-c00c2057bce7.png)



## RESULT

Thus,the neural network regression model for the given dataset is developed.
