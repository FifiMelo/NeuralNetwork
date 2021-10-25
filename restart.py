import pickle
import random

if False:
    structure = [784,20,20,10]
    pkl_file = open('data.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    print(data1)
    pkl_file.close()
    for layer in range(len(structure) - 1):
        for Lneuron in range(structure[layer + 1]):
            for L_1neuron in range(structure[layer]):
                data1["weights"][layer][L_1neuron][Lneuron] = (random.random()*2)-1
            data1["biases"][layer][Lneuron] = (random.random()*2)-1
    data1["x"] = 0
    data1["epoch"] = 0
    #data1["indexes"] = list(range(60000))
    output = open('data.pkl', 'wb')
    pickle.dump(data1, output)
    output.close()
    print("data restarted")
    exit()
