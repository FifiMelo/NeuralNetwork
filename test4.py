import pickle
import AI_sierpien2k21 as ai
import data
import matplotlib.pyplot as plt
import numpy as np
import time

pkl_file = open('result3.pkl', 'rb')
data1 = pickle.load(pkl_file)
pkl_file.close()


myAI = ai.NN([28*28,20,20,10])
myAI.weights = data1["weights"]
myAI.biases = data1["biases"]

scanner = data.Input()
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



answer = [0,0,0,0,0,0,0,0,0,0]
cost_sum = 0
for index in range(59000,60000):
    start_time = time.time()
    myAI.train(scanner.get_input(index),answer)
    print((time.time() - start_time)*1000)
    correct_answer = [0 for x in range(10)]
    correct_answer[scanner.get_answer(index,True)] = 1
    cost_sum += ai.calculate_cost(answer,correct_answer)
    
    time.sleep(0.02)
    #print(index)

print(cost_sum/10000)

plt.imshow(np.array(scanner.get_input(index,False)).reshape((28,28)))
plt.show()




