import AI_sierpien2k21 as AI
import time
import pickle
import numpy as np
import data

batch_size = 32
if 60000%batch_size:
    print("error, put batch size, that 60 000 is divisible by")
scanner = data.Scanner()
scanner.begin_stream()
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
print(actual_x)
scanner.start_reading_data_from_index(actual_x)
scanner.start_reading_labels_from_index(actual_x)
try:
    while actual_epochs < 5:
        while actual_x < 60000:
            answer = [0 for x in range(10)]
            answer[scanner.read_label()] = 1
            myAI.train(scanner.read_data(),answer)
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
            answer = [0 for x in range(10)]
            correct_answer = scanner.read_label(False)
            answer[correct_answer] = 1
            predict = myAI.give_answer(scanner.read_data(False))
            sum+=AI.calculate_cost(predict,answer)
            if not np.argmax(predict) == correct_answer:
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
        answer = [0 for x in range(10)]
        answer[scanner.read_label()] = 1
        myAI.train(scanner.read_data(),answer)
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
    scanner.close_stream()

