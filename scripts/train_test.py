import  numpy
import  matplotlib.pyplot as plt
numpy.random.seed(2)

x=numpy.random.normal(3,1,100)
y=numpy.random.normal(150,40,100)/x
plt.scatter(x,y)
plt.show()

# hand rule: 80% train, 20% test

train_x= x[:80]
train_y= y[:80]

test_x= x[80:]
test_y= y[80:]

plt.scatter(train_x,train_y)
plt.show()

plt.scatter(test_x,test_y)
plt.show()