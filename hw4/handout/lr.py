import numpy as np
import argparse
import matplotlib.pyplot as plt
import math


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def plot(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    val: np.ndarray,
    val_y: np.ndarray,
    num_epoch : int, 
    learning_rate : float
) -> None:
    y_1_axis = np.array([])
    y_10_axis = np.array([])
    y_100_axis = np.array([])
    x_axis = np.array([])
    # val_line = np.array([])

    theta_1 = np.zeros(301)
    theta_10 = np.zeros(301)
    theta_100 = np.zeros(301)
    for x in range(0,num_epoch):
        x_axis = np.append(x_axis,x)
        for i in range(0,X.shape[0]):
            coef = sigmoid(X[i]@theta_1)-y[i]
            grad = np.dot((X[i].transpose()),coef)
            theta_1 -= 0.1 * grad
        
        for i in range(0,X.shape[0]):
            coef = sigmoid(X[i]@theta_10)-y[i]
            grad = np.dot((X[i].transpose()),coef)
            theta_10 -= 0.01 * grad
        
        for i in range(0,X.shape[0]):
            coef = sigmoid(X[i]@theta_100)-y[i]
            grad = np.dot((X[i].transpose()),coef)
            theta_100 -= 0.001 * grad
        
        g = 0
        for i in range(0,X.shape[0]):
            g += y[i]*math.log(sigmoid(theta_1@X[i].T))+(1-y[i])*math.log(1-sigmoid(theta_1@X[i].T))
        g /= X.shape[0]
        g = -g
        y_1_axis = np.append(y_1_axis,g)

        g = 0
        for i in range(0,X.shape[0]):
            g += y[i]*math.log(sigmoid(theta_10@X[i].T))+(1-y[i])*math.log(1-sigmoid(theta_10@X[i].T))
        g /= X.shape[0]
        g = -g
        y_10_axis = np.append(y_10_axis,g)

        g = 0
        for i in range(0,X.shape[0]):
            g += y[i]*math.log(sigmoid(theta_100@X[i].T))+(1-y[i])*math.log(1-sigmoid(theta_100@X[i].T))
        g /= X.shape[0]
        g = -g
        y_100_axis = np.append(y_100_axis,g)
        

        # g = 0
        # for i in range(0,val.shape[0]):
        #     g += val_y[i]*math.log(sigmoid(theta@val[i].T))+(1-val_y[i])*math.log(1-sigmoid(theta@val[i].T))
        # g /= val.shape[0]
        # g = -g
        # val_line = np.append(val_line,g)
    
    plt.plot(x_axis, y_1_axis,label='rate=0.1',color='blue')
    plt.plot(x_axis,y_10_axis,label='rate=0.01',color='red')
    plt.plot(x_axis,y_100_axis,label='rate=0.001',color='green')
    plt.title('Average Negative Log-likelihood')
    plt.xlabel('Number of epochs')
    plt.ylabel('Negative Log-likelihood')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> np.ndarray:
    # TODO: Implement `train` using vectorization
    y_axis = np.array([])
    x_axis = np.array([])
    size = X.shape[0]
    for x in range(0,num_epoch):
        g = 0
        for i in range(0,size):
            coef = sigmoid(X[i]@theta)-y[i]
            grad = np.dot((X[i].transpose()),coef)
            theta -= learning_rate * grad
            g=np.mean(grad)
        y_axis = np.append(y_axis,g)
        x_axis = np.append(x_axis,x)
    return theta

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    res = np.zeros(X.shape[0])
    for i in range(0,X.shape[0]):
        if theta@X[i] < 0:
            res[i] = 0
        else:
            res[i] = 1
    return res


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    cor = 0
    for i in range(0,y.size):
        if(y[i]==y_pred[i]):
            cor +=1
    return 1-cor/y.size

def process(data_file):
    labels = np.array([])
    elements = np.zeros((1,301))
    with open(data_file,'r') as file:
        for x in file:
            list = x.split()
            labels = np.append(labels,float(list[0]))
            add = np.append(np.array([1]),np.array([float(i) for i in list[1:]]))
            elements = np.append(elements,np.array([add]),axis=0)
    return labels, elements[1:]

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    t1 = np.zeros(4)
    # print(t1)
    # print(t1.size)
    # t1 = sigmoid(t1)
    # print(t1)
    # print(t1.size)
    # print((np.array([1,2,3]))@np.array([1,2,3]))

    train_y, train_data = process(args.train_input)
    val_y,val_data = process(args.validation_input)
    theta = np.zeros(301)
    plot(theta,train_data,train_y,val_data,val_y,args.num_epoch,args.learning_rate)
    grad_t = train(theta,train_data,train_y,args.num_epoch,args.learning_rate)
    train_predict = predict(grad_t,train_data)
    train_err = compute_error(train_predict,train_y)
    print(train_err)

    test_y, test_data = process(args.test_input)
    test_predict = predict(grad_t,test_data)
    test_err = compute_error(test_predict,test_y)
    print(test_err)

    print(train_data)

    train_out = ""
    for x in train_predict:
        train_out += str(x)+"\n"
    train_out = train_out[:-1]
    with open(args.train_out,'w') as file:
        file.write(train_out)
    
    test_out = ""
    for x in test_predict:
        test_out += str(x)+"\n"
    test_out = test_out[:-1]
    with open(args.test_out,'w') as file:
        file.write(test_out)

    metric_s = f"error(train): {round(train_err,6)}\nerror(test): {round(test_err,6)}"
    with open(args.metrics_out,'w') as file:
        file.write(metric_s)
