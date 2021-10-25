import numpy as np
import data
import random
import time



def sigmoid(x):
    if x>=0:
        return 1 / (1 + np.exp(-x))
    else:
        return 1 - 1/(1  + np.exp(x))

def vector_sigmoid(vector):
    return [sigmoid(x) for x in vector]

def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def calculate_cost(tab1,tab2):
    suma = 0
    for x,y in zip(tab1, tab2):
        suma += (x - y)*(x - y)
    return suma

def shuffle(tab):
    return [tab.pop(random.randint(0,len(tab)-1)) for x in range(len(tab))]

class NegativeGradient:
    def __init__(self,structure):
        self.memory_weights = []
        self.memory_biases = []
        self.weights = []
        self.biases = []
        self.structure = structure
        for new_layer_index in range(len(structure) - 1):
            self.weights.append(np.zeros((structure[new_layer_index], structure[new_layer_index + 1])))
            self.biases.append(np.zeros(structure[new_layer_index + 1]))
            self.memory_weights.append(np.zeros((structure[new_layer_index], structure[new_layer_index + 1])))
            self.memory_biases.append(np.zeros(structure[new_layer_index + 1]))
        self.trening_examples_number = 0

    def assign(self):
        for layer in range(len(self.structure) - 1):
            for Lneuron in range(self.structure[layer + 1]):
                for L_1neuron in range(self.structure[layer]):
                    self.memory_weights[layer][L_1neuron][Lneuron] = self.memory_weights[layer][L_1neuron][Lneuron]*(self.trening_examples_number/(self.trening_examples_number + 1)) + self.weights[layer][L_1neuron][Lneuron]/(self.trening_examples_number + 1)
                self.memory_biases[layer][Lneuron] = self.memory_biases[layer][Lneuron]*(self.trening_examples_number/(self.trening_examples_number + 1)) + self.biases[layer][Lneuron]/(self.trening_examples_number + 1)
        self.trening_examples_number += 1   

                          

class NN:
    def __init__(self, structure):
        self.structure = structure
        self.layers_number = len(structure)
        self.weights = []
        self.biases = []
        for new_layer_index in range(self.layers_number - 1):
            self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(structure[new_layer_index], structure[new_layer_index + 1])))
            self.biases.append(np.random.uniform(low=-1.0, high=1.0, size=(structure[new_layer_index + 1])))
        self.gradient = NegativeGradient(structure)
        



    def give_answer(self, input): 
        actual_computation = input
        for x in range(self.layers_number - 1):
            #start_time = time.time()
            actual_computation = np.dot(actual_computation, self.weights[x])
            #print(int((time.time() - start_time)*1000))
            actual_computation = np.add(actual_computation, self.biases[x])
            actual_computation = vector_sigmoid(actual_computation)
        return(actual_computation)



    def train(self,input, answer, learn_speed = 1.0):
        zees = []
        neurons = []
        chain = []
        actual_computation = input
        zees.append(input)
        neurons.append(input)
        #chain.append(np.zeros(shape=(self.structure[0])))
        for x in range(self.layers_number - 1):
            actual_computation = np.dot(actual_computation, self.weights[x])
            actual_computation = np.add(actual_computation, self.biases[x])
            zees.append(actual_computation)
            actual_computation = vector_sigmoid(actual_computation)
            chain.append(np.zeros(shape=(self.structure[x+1])))
            neurons.append(actual_computation)
        for x in range(self.structure[self.layers_number - 1]):
            chain[self.layers_number-2][x] = 2*(neurons[self.layers_number - 1][x] - answer[x])
        for layer in range(self.layers_number - 1)[::-1]:
            for Lneuron in range(self.structure[layer + 1]):
                self.gradient.biases[layer][Lneuron] = chain[layer][Lneuron]*sigmoid_derivative(zees[layer+1][Lneuron])*learn_speed
                for L_1neuron in range(self.structure[layer]):
                    self.gradient.weights[layer][L_1neuron][Lneuron] = neurons[layer][L_1neuron]*sigmoid_derivative(zees[layer+1][Lneuron])*chain[layer][Lneuron]*learn_speed
                    if layer > 0:
                        chain[layer - 1][L_1neuron]+=self.weights[layer][L_1neuron][Lneuron]*sigmoid_derivative(zees[layer+1][Lneuron])*chain[layer][Lneuron]              
        self.gradient.assign()
 
    def end_batch(self):
        for layer in range(len(self.structure) - 1):
            for Lneuron in range(self.structure[layer + 1]):
                for L_1neuron in range(self.structure[layer]):
                    self.weights[layer][L_1neuron][Lneuron]+=self.gradient.memory_weights[layer][L_1neuron][Lneuron]*(-1)
                    self.gradient.memory_weights[layer][L_1neuron][Lneuron] = 0
                self.biases[layer][Lneuron]+=self.gradient.memory_biases[layer][Lneuron]*(-1)
                self.gradient.memory_biases[layer][Lneuron] = 0
        self.gradient.trening_examples_number = 0

