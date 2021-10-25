import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

class Scaner:
    def load_data_from_source(traning = True):
        if traning:
            input_name = "data_source/train-images-idx3-ubyte.gz"
            label_file = "data_source/train-labels-idx1-ubyte.gz"
            output_name = "train_data.pkl"
            data_size = 60000
        else:
            input_name = "data_source/t10k-images-idx3-ubyte.gz"
            label_file = "data_source/t10k-labels-idx1-ubyte.gz"
            output_name = "test_data.pkl"
            data_size = 10000
        output = open(output_name, "wb")
        pictures = gzip.open(input_name, 'r')
        labels = gzip.open(label_file, 'r')
        pictures_to_save = []
        labels_to_save = []
        np.frombuffer(labels.read(8), dtype=np.uint8).astype(np.float32)
        np.frombuffer(pictures.read(16), dtype=np.uint8).astype(np.float32)   
        for x in range(data_size):
            pictures_to_save.append(np.frombuffer(pictures.read(784), dtype=np.uint8).astype(np.float32))
            labels_to_save.append(np.frombuffer(labels.read(1), dtype=np.uint8).astype(np.float32))
        result = {
            "pictures": pictures_to_save,
            "labels": labels_to_save,
        }
        pickle.dump(result, output)
        output.close()
        pictures.close()
        labels.close()
    def begin_reading_data(training = True):
        pass




if __name__ == "__main__":
    pass