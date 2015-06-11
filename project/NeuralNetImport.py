import numpy as np
from sklearn.datasets import load_digits 
digits = load_digits()


""" The activation function. """
def sigmoid(x):
    return 1/(1+np.exp(-x))

""" Derivative of the activation function"""
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def get_perm(greatest_num, number):
    return np.random.permutation(greatest_num)[0:number]


def training_input_digits(perm_array, size, div):
    return np.array([np.append((digits.images[perm_array[i]].flatten())/div,[1]) for i in range(size)])


"""Created a training solution for a list of numbers or capital letters"""
def create_training_soln(training_numbers,length):
    a = np.repeat(0,length,None)
    a = np.repeat([a], len(training_numbers), 0)
    for i in range(len(training_numbers)):
        if type(training_numbers[i]) == np.int64:
            a[i][training_numbers[i]] = 1
        if type(training_numbers[i]) == np.str_:
            a[i][ord(training_numbers[i])-65] = 1
    return a



class NN_training(object):
    
    def __init__(self, input_array, soln, innum, outnum, hidnum, iters, lr):
        self.input_array = input_array
        self.soln = soln
        #Number of hidden nodes
        self.hidnum = hidnum
        #Number of iterations through the training set
        self.iters = iters
        #Initalize WIJ weights (input to hidden)
        self.innum = innum
        self.outnum = outnum
        self.wij = np.random.uniform(-.5,0.5,(hidnum,innum + 1))
        #Initalize WJK weights (hidden to output)
        self.wjk = np.random.uniform(-0.5,0.5,(outnum,hidnum+1))
        #Set a learning rate
        self.lr = lr
    def train(self):
        iters = self.iters   
        for n in range(iters): 
            for i in range(len(self.input_array)):
                soln = self.soln[i]
                hidnum = self.hidnum
                outnum = self.outnum
                input_array = np.append(self.input_array[i],[1])
                #Find sum of weights x input array values for each hidden
                self.hidden_sums = (sum((input_array * self.wij).T)).T
                #Find outputs of hidden neurons; include bias
                self.hidden_out = np.append(sigmoid(self.hidden_sums),[1])
                #Find sums of weights x hidden outs for each neuron in output layer
                self.output_sums = (sum((self.hidden_out * self.wjk).T)).T
                #Find output of the outputs
                self.output_out = sigmoid(self.output_sums)
                self.E = self.output_out - soln
                 #Find delta values for each output
                self.output_deltas = self.E * sigmoid_prime(self.output_sums)
                 #Find delta values for each hidden
                self.hidden_deltas = sigmoid_prime(np.delete(self.hidden_out,[hidnum],None)) * sum((self.output_deltas *(np.delete(self.wjk, [hidnum], 1)).T).T)
                #Change weights
                self.wij = -self.lr * (self.hidden_deltas*(np.repeat([input_array],hidnum,0)).T).T + self.wij
                self.wjk = -self.lr * (self.output_deltas*(np.repeat([self.hidden_out],outnum,0)).T).T + self.wjk
        return (self.wij, self.wjk)
    

    
    
class NN_ask (object):
    """ Feed forward using final weights from training backpropagation """
    def __init__(self, input_array, wij, wjk):
        self.input_array = input_array
        self.wij = wij
        self.wjk = wjk
    def get_ans(self):
        wij = self.wij
        wjk = self.wjk
        soln = []
        for i in range(len(self.input_array)):
            input_array = np.append(self.input_array[i],[1])
            self.hidden_sums = (sum((input_array * wij).T)).T
            self.hidden_out = np.append(sigmoid(self.hidden_sums),[1]) 
            self.output_sums = (sum((self.hidden_out * wjk).T)).T
            self.output_out = sigmoid(self.output_sums)
            for i in range(len(self.output_out)):
                if self.output_out[i] == max(self.output_out):
                    a = i
                    soln.append(a)
        return soln
    
    
    
    
class NN_training_2(object):
    
    def __init__(self, input_array, soln, innum, outnum, hidnum1, hidnum2, iters, lr):
        self.input_array = input_array
        self.soln = soln
        #Number of hidden nodes
        self.hidnum1 = hidnum1
        self.hidnum2 = hidnum2
        #Number of iterations through the training set
        self.iters = iters
        #Initalize WIJ weights (input to hidden)
        self.innum = innum
        self.outnum = outnum
        self.wij = np.random.uniform(-.5,0.5,(hidnum1,innum+1))
        #Initalize WJK weights (hidden to output)
        self.wjk = np.random.uniform(-0.5,0.5,(hidnum2,hidnum1+1))
        self.wkl = np.random.uniform(-0.5,0.5,(outnum,hidnum2+1))
        #Set a learning rate
        self.lr = lr
    def train(self):
        iters = self.iters   
        for n in range(iters): 
            for i in range(len(self.input_array)):
                soln = self.soln[i]
                hidnum1 = self.hidnum1
                hidnum2 = self.hidnum2
                innum = self.innum
                outnum = self.outnum
                input_array = np.append(self.input_array[i],[1])
                #Find sum of weights x input array values for each hidden
                self.hidden1_sums = np.transpose(sum(np.transpose(input_array * self.wij))) 
                #Find outputs of hidden neurons
                self.hidden1_out = np.append(sigmoid(self.hidden1_sums),[1])
                #SAME FOR SECOND HIDDEN LAYER
                self.hidden2_sums = np.transpose(sum(np.transpose(self.hidden1_out * self.wjk)))
                #OUTS OF HIDDEN2
                self.hidden2_out = np.append(sigmoid(self.hidden2_sums),[1])
                #Find sums of weights x hidden outs for each neuron in output layer
                self.output_sums = np.transpose(sum(np.transpose(self.hidden2_out * self.wkl))) 
                #Find output of the outputs
                self.output_out = sigmoid(self.output_sums)
                self.E = self.output_out - soln
                 #Find delta values for each output
                self.output_deltas = self.E * sigmoid_prime(self.output_sums)
                 #Find delta values for each hidden
                self.hidden2_deltas = sigmoid_prime(np.delete(self.hidden2_sums,[hidnum2],None)) * sum(np.transpose(self.output_deltas * np.transpose(np.delete(self.wkl, [hidnum2], 1))))
                self.hidden1_deltas = sigmoid_prime(np.delete(self.hidden1_sums,[hidnum1],None)) * sum(np.transpose(self.hidden2_deltas * np.transpose(np.delete(self.wjk,[hidnum1],1))))
                #Change weights
                self.wij = -self.lr * (self.hidden1_deltas*(np.repeat([input_array], hidnum1,0)).T).T + self.wij
                self.wjk = -self.lr * (self.hidden2_deltas*(np.repeat([self.hidden1_out],hidnum2,0)).T).T + self.wjk
                self.wkl = -self.lr * (self.output_deltas*(np.repeat([self.hidden2_out],outnum,0)).T).T + self.wkl
        return (self.wij, self.wjk, self.wkl)
    
 


class NN_ask_2 (object):
    """ Feed forward using final weights from training backpropagation """
    def __init__(self, input_array, wij, wjk, wkl):
        self.input_array = input_array
        self.wij = wij
        self.wjk = wjk
        self.wkl = wkl
    def get_ans(self):
        wij = self.wij
        wjk = self.wjk
        wkl = self.wkl
        for i in range(len(self.input_array)):
            input_array = np.append(self.input_array[i],[1])
            self.hidden1_sums = np.transpose(sum(np.transpose(input_array * self.wij))) 
            self.hidden1_out = np.append(sigmoid(self.hidden1_sums),[1])
            self.hidden2_sums = np.transpose(sum(np.transpose(self.hidden1_out * self.wjk)))
            self.hidden2_out = np.append(sigmoid(self.hidden2_sums),[1])
            self.output_sums = np.transpose(sum(np.transpose(self.hidden2_out * self.wkl))) 
            self.output_out = sigmoid(self.output_sums)      
            for i in range(len(self.output_out)):
                if self.output_out[i] == max(self.output_out):
                    a = i
        return a

    
class NN_ask_morse(object):
    """ Feed forward using final weights from training backpropagation """
    def __init__(self, input_array, wij, wjk):
        self.input_array = input_array
        self.wij = wij
        self.wjk = wjk
    def get_ans(self):
        wij = self.wij
        wjk = self.wjk
        soln = []
        for h in range(len(self.input_array)):
            for i in range(len(self.input_array[h])):
                input_array = np.append(self.input_array[h][i],[1])
                self.hidden_sums = (sum((input_array * wij).T)).T
                self.hidden_out = np.append(sigmoid(self.hidden_sums),[1]) 
                self.output_sums = (sum((self.hidden_out * wjk).T)).T
                self.output_out = sigmoid(self.output_sums)
                for i in range(len(self.output_out)):
                    if self.output_out[i] == max(self.output_out):
                        a = i
                        soln.append(a + 65)
            soln.append(32)
        return soln