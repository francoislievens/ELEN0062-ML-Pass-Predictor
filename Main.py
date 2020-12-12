import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Dataset import Dataset
from Neural import Neural
from Forest import Forest
import time
import matplotlib.pyplot as plt
import tensorflow as tf

def neural_1():

    # Build the model:
    model = Neural()

    # Create the dataset structure:
    dataset = Dataset()
    # Import original training set
    #dataset.import_original_training(split_train=0.8, split_test=0.2, split_val=0)
    # Compute the pair form of the dataset
    #dataset.learning_set_builders()
    # Save in a file to speed up experiments
    #dataset.save_dataset()

    # Restore dataset from a file:
    dataset.restore_dataset()

    # give the dataset to the model:
    model.set_dataset(dataset)


    report = model.train(report=True)
    # Predict on the kaggle data
    predictions = model.model(dataset.pairs_test_x)
    pred = tf.reshape(predictions, (-1, 22))
    pred = tf.nn.softmax(pred, axis=1)

    pred_final = tf.argmax(pred, axis=1)
    pred_final = pred_final.numpy()
    pred_final += 1

    result = np.zeros(len(pred_final))
    # Manual accuracy:
    for i in range(0, len(pred_final)):
        print('{} - {}'.format(pred_final[i], model.dataset.original_test_y[i]))
        if pred_final[i] == model.dataset.original_test_y[i]:
            result[i] = 1
    print('final test accuracy: {}'.format(np.mean(result)))

    # Predict on the kaggle data
    predictions = model.model(dataset.final_pairs)
    pred = tf.reshape(predictions, (-1, 22))
    pred = tf.nn.softmax(pred, axis=1)

    pred_final = tf.argmax(pred, axis=1)
    pred_final = pred_final.numpy()
    pred_final += 1

    write_submission(predictions=pred_final, probas=pred.numpy())

    plt.plot(report[:, 0], report[:, 1], c='green', label='Test Loss')
    plt.plot(report[:, 0], report[:, 2], c='red', label='Train Loss')
    plt.plot(report[:, 0], report[:, 3], c='blue', label='Test Loss pass predict')
    plt.plot(report[:, 0], report[:, 4], c='orange', label='Train Loss pass predict')
    plt.plot(report[:, 0], report[:, 5], c='blue', label='Test Accuracy pass predict')
    plt.plot(report[:, 0], report[:, 6], c='orange', label='Train Accuracy pass predict')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def random_forest_1():

    # Create the dataset structure:
    dataset = Dataset()
    # Import original training set
    dataset.import_original_training(split_train=0.8, split_test=0.17, split_val=0.03)
    # Compute the pair form of the dataset
    dataset.learning_set_builders()
    # Save in a file to speed up experiments
    dataset.save_dataset()

    # Restore dataset from a file:
    dataset.restore_dataset()

    # Create the model
    model = Forest()
    # Set the dataset
    model.set_dataset(dataset)
    # Train the model
    model.train()

def write_submission(predictions=None, probas=None, estimated_score=0.375, file_name="Original_data/submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    predictions: array [n_predictions, 1]
        `predictions[i]` is the prediction for player
        receiving pass `i` (or indexes[i] if given).
    probas: array [n_predictions, 22]
        `probas[i,j]` is the probability that player `j` receives
        the ball with pass `i`.
    estimated_score: float [1]
        The estimated accuracy of predictions.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    if predictions is None and probas is None:
        raise ValueError('Predictions and/or probas should be provided.')

    n_samples = len(predictions)
    if indexes is None:
        indexes = np.arange(n_samples)

    if probas is None:
        print('Deriving probabilities from predictions.')
        probas = np.zeros((n_samples,22))
        for i in range(n_samples):
            probas[i, predictions[i]-1] = 1

    if predictions is None:
        print('Deriving predictions from probabilities')
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            mask = probas[i] == np.max(probas[i])
            selected_players = np.arange(1,23)[mask]
            predictions[i] = int(selected_players[0])


    # Writing into the file
    with open(file_name, 'w') as handle:
        # Creating header
        header = '"Id","Predicted",'
        for j in range(1,23):
            header = header + '"P_{:0.0f}",'.format(j)
        handle.write(header[:-1]+"\n")

        # Adding your estimated score
        first_line = '"Estimation",{},'.format(estimated_score)
        for j in range(1,23):
            first_line = first_line + '0,'
        handle.write(first_line[:-1]+"\n")

        # Adding your predictions
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i])
            pj = probas[i, :]
            for j in range(22):
                line = line + '{},'.format(pj[j])
            handle.write(line[:-1]+"\n")

    return file_name

if __name__ == '__main__':

    neural_1()

    #random_forest_1()











