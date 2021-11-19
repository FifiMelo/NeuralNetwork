import AI_sierpien2k21 as ai
import numpy as np
import random

input_size = 10000
epochs = 5
model = ai.NN([2,1])

train_data = []
train_labels = []
#model.weights = [np.array([[-10.0], [-10.0]])]
#model.biases = [np.array([15.0])]
print("weights:")
print(model.weights)
print("biases:")
print(model.biases)
print("_________________________")



for x in range(input_size):
    a = random.getrandbits(1)
    b = random.getrandbits(1)
    train_data.append([a,b])
    if not (a and b):
        train_labels.append([1])
    else:
        train_labels.append([0])

cost = 0
for index, label, data in zip(range(input_size), train_labels, train_data):
    cost+= ai.calculate_cost(model.give_answer(data), label)
print(f"cost:{cost/input_size}")

for x in range(epochs):
    for index, label, data in zip(range(1, input_size+1), train_labels, train_data):
        model.train(data, label)
        if not index%5:
            model.end_batch()

cost = 0
for index, label, data in zip(range(input_size), train_labels, train_data):
    cost+= ai.calculate_cost(model.give_answer(data), label)


print(model.give_answer([0,0]))
print(model.give_answer([1,0]))
print(model.give_answer([0,1]))
print(model.give_answer([1,1]))
print("_________________________")
print("weights:")
print(model.weights)
print("biases:")
print(model.biases)
print(f"cost:{cost/input_size}")
