import pickle
import AI_sierpien2k21 as ai
import data
import matplotlib.pyplot as plt
import numpy as np
import time
import math as m



pkl_file = open('result3.pkl', 'rb')
data1 = pickle.load(pkl_file)
pkl_file.close()


myAI = ai.NN([28*28,20,20,10])
myAI.weights = data1["weights"]
myAI.biases = data1["biases"]

scanner = data.Scanner()
"""
index = 0
answer = myAI.give_answer(scanner.get_input(index,False))
correct_answer = [0 for x in range(10)]
correct_answer[scanner.get_answer(index,False)] = 1
print(f"The answer is {answer}")
print(f"The corrext answer is {scanner.get_answer(index,False)}")
print(f"Cost is equal to {ai.calculate_cost(answer,correct_answer)}")
"""
random_image = np.random.randint(256, size=(784))

scanner.begin_stream()
"""
#answer = [0 for x in range(10)]
false_predicts = 0
cost_sum = 0
for index in range(0,60000):
    start_time = time.time()
    answer = myAI.give_answer(scanner.read_data(True))
    #print((time.time() - start_time)*1000)
    correct_answer = [0 for x in range(10)]
    correct_index = scanner.read_label(True)
    correct_answer[correct_index] = 1
    cost_sum += ai.calculate_cost(answer,correct_answer)
    if not np.argmax(answer) == correct_index:
        false_predicts += 1
    print(index//600)
print(cost_sum/60000)
print(f"False predicts: {false_predicts}")
"""
average1 = 0.10640368660450422
average2 = 0
sigma = 0
for x in range(10000):
    for a in myAI.give_answer(scanner.read_data(True)):
        average2 += a
        sigma += abs(a - average1)
    print(x//100)
print(average2/(10*10000))
print((sigma/(10*10000)))
scanner.close_stream()

