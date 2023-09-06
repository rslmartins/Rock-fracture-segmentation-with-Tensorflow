# coding: utf-8

from network import Net
from trainer import Trainer
import numpy as np
import argparse


network_type = 5
first_filter = 64
class_weight = [0.7, 0.3]
norm_type = "batch_norm"
epochs = 15
learning_rate = 0.001
batch_size = 5
keep_prob = 0.8
model_path = "models/model.ckpt"


def train():
	x_tr = np.load('./x_train.npy')
	y_tr = np.load('./y_train.npy')
	network = Net(network_type, 224, first_filter, class_weights=class_weight, norm_kwargs={'norm_type': norm_type})
	trainer = Trainer(network, x_tr, y_tr)

	trainer.training(epochs, learning_rate, batch_size, keep_prob, None, model_path)

def test():
	x_test = np.load("./x_test.npy")
	y_test = np.load("./y_test.npy")
	network = Net(network_type, 1024, first_filter, class_weights=class_weight, norm_kwargs={'norm_type': norm_type})
	trainer = Trainer(network)
        
	trainer.evaluate(x_test, y_test, f"{model_path}-ep_{epochs-1}")
	softmax, prediction = trainer.get_output(x_test, f"{model_path}-ep_{epochs-1}")
	np.save("softmax_output.npy", softmax)

def flag():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your script.')
    # Add the -i flag with a help message
    parser.add_argument('-i', '--input', help='Input argument description')
    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the value of the -i flag (if provided)
    input_value = args.input
    # Your code logic goes here, and you can use 'input_value' as needed
    if input_value == "test":
        test()
    elif input_value == "train":
        train()
    else:
        print('Invalid input value. Please use "test" or "train".')

if __name__ == "__main__":
    flag()
