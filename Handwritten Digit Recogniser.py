from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
##Internet required
mnist=fetch_openml('mnist_784',version=1)
#mnist=fetch_openml()
print(mnist.keys())
print(mnist['DESCR'])

X=mnist['data']
y=mnist['target']

print(X)
print(y)

print(X.shape)
print(y.shape)

import numpy as np
y=np.uint8(y)

##Splitting the data into Training data and Test data
X_train=X[:60000]
X_test=X[60000:]
y_train=y[:60000]
y_test=y[60000:]

import matplotlib.pyplot as plt
# We need to modify the color map of the matplot
plt.imshow(X[0].reshape(28,28))
plt.show()

import matplotlib as mpl
##cmap is colormap
## 
plt.imshow(X[0].reshape(28,28),cmap=mpl.cm.binary)
plt.show()
# Display function
def show_digits(m,n):
    fig, im=plt.subplots(m,n)
    for i in range(m):
        for j in range(n):
            im[i,j].imshow(X[i+j].reshape(28,28),cmap=mpl.cm.binary)
            im[i,j].set_title(y[i+j])
    plt.show()

# show_digits(4,4)

sgd=SGDClassifier(random_state=42)
##sgd.fit(X_train,y_train)
acc=[]
def pred_digit(x):
    y_train_x=(y_train==x)
    y_test_x=(y_test==x)
    sgd.fit(X_train,y_train_x)
    pred_test_x=sgd.predict(X_test)
    accuracy=(sum(pred_test_x==y_test_x)/len(y_test_x))*100
    print(x,": ",accuracy)
    acc.append(accuracy)

for i in range(10):
    pred_digit(i)

x=[item for item in range(0,10)]
plt.bar(x,acc)
plt.ylim(90,100)
plt.title('Digits VS Accuracy')
plt.show()
