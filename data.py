import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt



class Scanner:
    def begin_stream(self):
        self.training_data_file = gzip.open("data_source/train-images-idx3-ubyte.gz")
        self.training_labels_file = gzip.open("data_source/train-labels-idx1-ubyte.gz")
        self.testing_data_file = gzip.open("data_source/t10k-images-idx3-ubyte.gz")
        self.testing_labels_file = gzip.open("data_source/t10k-labels-idx1-ubyte.gz")
        
        self.training_data_file.read(16)
        self.training_labels_file.read(8)
        self.testing_data_file.read(16)
        self.testing_labels_file.read(8)

    def read_data(self,training = True):
        if training:
            return np.frombuffer(self.training_data_file.read(784), dtype=np.uint8)
        else:
            return np.frombuffer(self.testing_data_file.read(784), dtype=np.uint8)

    def start_reading_data_from_index(self, index, training = True):
        if training:
            np.frombuffer(self.training_data_file.read(784*index), dtype=np.uint8)
        else:
            np.frombuffer(self.testing_data_file.read(784*index), dtype=np.uint8)

    def read_label(self,training = True):
        if training:
            return np.frombuffer(self.training_labels_file.read(1), dtype=np.uint8)[0]
        
        else:
            return np.frombuffer(self.testing_labels_file.read(1), dtype=np.uint8)[0]

    
    def start_reading_labels_from_index(self, index, training = True):
        if training:
            np.frombuffer(self.training_labels_file.read(index), dtype=np.uint8)
        else:
            np.frombuffer(self.testing_labels_file.read(index), dtype=np.uint8)

    def close_stream(self):
        self.training_data_file.close()
        self.training_labels_file.close()
        self.testing_data_file.close()
        self.testing_labels_file.close()


if __name__ == "__main__":
    pass


