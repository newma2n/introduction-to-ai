import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import scipy
import scipy.sparse
import inspect
import sys

if (__name__ == "__main__") :

    # Check arguments
    if len(sys.argv) != 2 :
        print("Usage: python3", sys.argv[0], "<path_to_pyrat_dataset>")
        exit()
    dataset_name = sys.argv[1]

    ### This files reloads the pyrat_dataset that was stored as a pkl file by the generate dataset script. 
    x,y, mazeWidth, mazeHeight = pickle.load(open(dataset_name,"rb"))


    ## As the dataset was stored using scipy sparse array to save space, we convert it back to torch dense array. 
    ## Note that you could keep the sparse representation if you work with a machine learning method that accepts sparse arrays. 
    x = scipy.sparse.vstack(x).todense()
    y = scipy.sparse.vstack(y).todense()
    x = torch.FloatTensor(x).reshape(-1,(2*mazeHeight-1)*(2*mazeWidth-1))  # (number of moves, size of the canvas)
    y = torch.argmax(torch.FloatTensor(y), dim=1)  # (number of moves,)
    canvas_size = x.shape[1]
    print(x.shape, y.shape)

    ### Now you have to train a classifier using supervised learning and evaluate it's performance. 
    ### Let's try a neural network.

    ## Split your data into x_train, x_test, y_train, y_test.
    n = int(x.shape[0] * 80/100)  # number of examples in the train set
    x_train = x[:n]
    x_test = x[n:]
    y_train = y[:n]
    y_test = y[n:]

    ########
    # TODO #    Change the network model to fit your needs
    ########

    ## To begin with, define a neural network with two hidden layers. In pytorch, this correspond to only adding two layers of type "Linear".
    ## You need to make sure that the size of the input of the first layer correspond to the width of your X vector. 
    ## Feel free to try different number of layer and other non linear function.
    class Net(nn.Module):
        def __init__(self, in_features):
            super(Net, self).__init__()
            ### To complete

        def forward(self, x):
            ### To complete
            return x

    net = Net(canvas_size)

    ########
    # TODO #    Define a loss function and optimizer
    ########

    criterion = None
    optimizer = None

    ########
    # TODO #    Train the network
    ########

    n_epoch = 0 ### Number of epoch To complete
    n_batch = 0 ### Number of batch To complete
    n_per_batch = int(n / n_batch)  # number of examples per batch

    for epoch in range(n_epoch):
        running_loss = 0
        for b in range(n_batch):
            # get the inputs
            inputs = x_train[b*n_per_batch:b*n_per_batch+n_per_batch, :] 
            labels = y_train[b*n_per_batch:b*n_per_batch+n_per_batch]
            # zero the parameter gradients
            ### To complete
            # forward + loss + backward + optimize
            ### To complete
            ### To complete
            ### To complete
            ### To complete
            # statistics
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / n_batch))
        running_loss = 0.0

    print('Finished Training')

    ## Training accuracy
    correct = 0
    total = n
    with torch.no_grad():
        outputs = net(x_train)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_train).sum().item()

    print('Training accuracy of the network: %d %%' % (
        100 * correct / total))
        
    ########
    # TODO #    Test accuracy
    ########

    ### To complete

    ## Save the weights
    pickle.dump([inspect.getsource(Net), net.state_dict()], open("trained_classifier.pkl","wb"))



    ### Remarks
    ## If the training accuracy is about 25%, it means the network predicts the result
    ## as good as chance (4 possible choices).

    ## When you train a neural network, you have to analyze your results.
    ## If, after the training, your training accuracy is far from 100%, your network is underfitting (high bias).
    ## Try to train the network longer (more epochs, bigger/smaller learning rate, batch size).
    ## Or, define a bigger network (more hidden layers, bigger out_features).
    ## If, your test accuracy is far from your training accuracy, your network is overfitting (high variance).
    ## Try to regularize your optimization (look at L2 regularization, weight decay, drop out, early stopping...).
    ## Try to use more data.
