import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Dataset import Dataset
import Features_computers
from Forest import Forest
import pickle

if __name__ == '__main__':


    # Create the dataset structure:
    dataset = Dataset()
    # Import original training set
    #dataset.import_original_training(split_train=0.8, split_test=0.17, split_val=0.03)
    # Compute the pair form of the dataset
    #dataset.learning_set_builders()
    # Save in a file to speed up experiments
    #dataset.save_dataset()

    # Restore dataset from a file:
    dataset.restore_dataset()

    # Create the model
    model = Forest()
    # Set the dataset
    model.set_dataset(dataset)
    # Train the model
    model.train()

    # Serialize the model:
    model.save_model()

    # Restore
    #model.restore_model()

    # predict
    pred = model.rf.predict(dataset.pairs_test_x)
    pred = np.reshape(pred, (-1, 22))
    pred_idx = np.argmax(pred, axis=1)
    pred_idx += 1

    score = np.zeros(pred_idx.shape[0])
    for i in range(0, len(pred_idx)):
        print('pred: {} - targets: {}'.format(pred_idx[i], dataset.original_test_y[i]))
        if pred_idx[i] == dataset.original_test_y[i]:
            score[i] = 1
    print('final score = {}'.format(np.mean(score)))


