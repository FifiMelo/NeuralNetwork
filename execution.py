import AI_sierpien2k21 as AI
import data
import time
import pickle
import numpy as np

batch_size = 32
if 60000%batch_size:
    print("error, put batch size, that 60 000 is divisible by")
scanner = data.Input()
myAI = AI.NN([784,20,20,10])

pkl_file = open('data.pkl', 'rb')
data1 = pickle.load(pkl_file)
pkl_file.close()

actual_x = data1["x"]
actual_epochs = data1["epoch"]
myAI.weights = data1["weights"]
myAI.biases = data1["biases"]
#indexes = data1["indexes"]
print("actual_x: " + str(actual_x))

try:
    while actual_epochs < 5:
        while actual_x < 60000:
            answer = [0,0,0,0,0,0,0,0,0,0]
            answer[scanner.get_answer(actual_x)] = 1
            myAI.train(scanner.get_input(actual_x),answer, 1.5)
            if actual_x%batch_size == batch_size-1:
                myAI.end_batch()
            actual_x += 1        
        actual_epochs += 1
        print("epoch "+ str(actual_epochs) + " is completed")
        print("cost is being calculated, do not disturb...")
        actual_x = 0
        sum = 0.0
        false_predicts = 0
        for m in range(10000):
            answer = [0,0,0,0,0,0,0,0,0,0]
            answer[scanner.get_answer(m,False)] = 1
            predict = myAI.give_answer(scanner.get_input(m,False))
            sum+=AI.calculate_cost(predict,answer)
            if not np.argmax(predict) == scanner.get_answer(m,False):
                false_predicts += 1 
        print("average cost is: " + str(sum/10000))
        print("false predictions number: " + str(false_predicts))
        print("accuracy: " + str((100 - float(false_predicts)/100)) + "%")
        result = {
            "weights": myAI.weights,
            "biases": myAI.biases,
        }
        output = open('result' + str(actual_epochs) + '.pkl', 'wb')
        pickle.dump(result, output)
        output.close()
except KeyboardInterrupt:
    print("finishing this batch...")
    while actual_x % batch_size:
        answer = [0,0,0,0,0,0,0,0,0,0]
        answer[scanner.get_answer(actual_x)] = 1
        myAI.train(scanner.get_input(actual_x),answer)
        actual_x+=1
    myAI.end_batch()    
    print("saving...")
    print("actual_x: " + str(actual_x))
    info = {
        "weights": myAI.weights,
        "biases": myAI.biases,
        "x": actual_x,
        "epoch": actual_epochs,
        #"indexes": indexes
        }
    output = open('data.pkl', 'wb')
    pickle.dump(info, output)
    output.close()
    exit()

sum = 0.0
for m in range(10000):
    answer = [0,0,0,0,0,0,0,0,0,0]
    answer[scanner.get_answer(x,False)] = 1
    sum+=AI.calculate_cost(myAI.give_answer(scanner.get_input(m,False)),answer)
print("average cost is: " + str(sum/10000))
info = {
        "weights": myAI.weights,
        "biases": myAI.biases,
        "x": actual_x,
        "epoch": actual_epochs,
        #"indexes": indexes
        }
output = open('data.pkl', 'wb')
pickle.dump(info, output)
output.close()
exit()