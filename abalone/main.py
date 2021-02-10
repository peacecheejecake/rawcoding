import numpy as np
import csv
import os.path

from utils import read_csv, one_hot
from utils import DataIterator
from models import SingleLayerPerceptron



# hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 10

TEST_RATIO = 0.1

# data chracteristics
output_cols = 1
data_path = ['..', 'data']
data_file_name = 'abalone.csv'

# report settings
report_every = 1
test_size = 50
title_width = 40


def load_abalone_dataset(test_ratio=TEST_RATIO):
    raw_data = read_csv(data_path, data_file_name, False)
    data = one_hot(raw_data, [0]).astype(np.float64)
    test_size = int(data.shape[0] * test_ratio)
    return data[:-test_size], data[-test_size:]


if __name__ == "__main__":
    # data arrangement
    train_data, test_data = load_abalone_dataset(test_ratio=TEST_RATIO)
    train_iter = DataIterator(data=train_data, output_dim=output_cols, \
        batch_size=BATCH_SIZE)
    test_iter  = DataIterator(data=test_data, output_dim=output_cols, \
        batch_size=BATCH_SIZE)

    # create a model & initialize parameters
    Model = SingleLayerPerceptron(name='slp', xdim=train_iter.xdim, ydim=train_iter.ydim)
    Model.init_parameters()

    # first evaluation before training
    print(" Initial Evaluation Accuracy ".center(title_width, '='))
    print(f"Train Accuracy: [{Model.evaluate(train_iter)}], \
Test Accuracy: [{Model.evaluate(test_iter)}]", end='\n\n')

    # train
    print(" Start of Training ".center(title_width, '='))
    for epoch in range(1, EPOCHS + 1):
        loss_epoch_sum = 0
        for batch_in, batch_out in train_iter:
            loss_epoch_sum += Model.backward(batch_in, batch_out, LEARNING_RATE)
        loss = loss_epoch_sum / len(train_data)

        train_accuracy = Model.evaluate(train_iter)
        test_accuracy = Model.evaluate(test_iter)
        if epoch % report_every == 0 and epoch != EPOCHS:
            print(f"Epoch[{str(epoch).rjust(2)}] --- Loss: [{loss:.3f}],", \
                  f"Train Accuracy: [{train_accuracy:.3f}]," \
                  f"Test Accuracy: [{test_accuracy:.3f}]")

    print("End of Training.", end='\n\n')


    # final test
    print(" Test ".center(title_width, '='))

    batch_in, batch_out = next(test_iter._reset(test_size))
    predictions = Model.forward(batch_in)
    correct_count = 0
    for prediction, answer in zip(np.round(predictions), batch_out.astype(np.int64)):
        if prediction == answer:
            is_correct = "Correct!"
            correct_count += 1
        else:
            is_correct = "Wrong!"
        print(f"{is_correct.ljust(8)}   Predicted: {str(prediction.item()).rjust(2)}, \
Answer: {str(answer.item()).rjust(2)}")
    print(f"> Correct Ratio:  [{correct_count / test_size:.5f}]")
    print(f"  Train Accuracy: [{Model.evaluate(train_iter):.5f}], \
Test Accuracy: [{Model.evaluate(test_iter):.5f}]", end='\n\n')


    # weights and bias
    print(" Parameters ".center(title_width, '='))
    print(f"Weights: {Model.linear.weight.reshape(-1)}")
    print(f"Bias: {Model.linear.bias.reshape(-1)}")
    print()