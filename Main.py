import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Dataset import Dataset
from Neural import Neural
import Features_computers
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # Build the model:
    model = Neural()

    # Create the dataset structure:
    dataset = Dataset()
    # Import original training set
    dataset.import_original_training(split_train=0.7, split_test=0.27, split_val=0.03)
    # Compute the pair form of the dataset
    dataset.learning_set_builders()
    # Save in a file to speed up experiments
    dataset.save_dataset()

    # Restore dataset from a file:
    dataset.restore_dataset()

    test_x, test_y = dataset.make_players_pairs(pd.DataFrame(dataset.original_test_x, columns=dataset.original_x_header),
                                                             pd.DataFrame(dataset.original_test_y, columns=dataset.original_y_header))

    # give the dataset to the model:
    model.set_dataset(dataset)
    # Train the model
    model.train()

    # Test the model:
    # make predictions on test set:
    df = pd.DataFrame(model.dataset.original_test_x, columns=model.dataset.original_x_header)
    pred = model.predict_pass(df)
    # Compare with output:
    result = np.zeros(len(pred))
    for i in range(0, len(pred)):
        if i < 40:
            print('predicted: {} - truth: {}'.format(pred[i], model.dataset.original_test_y[i]))
        if pred[i] == model.dataset.original_test_y[i]:
            result[i] = 1
    print('final result: ')
    print(np.mean(result))

    temp_x = pd.read_csv('personal_data/original_test_x.csv', sep=',', index_col="Id")
    print(temp_x.head())
    index = 2
    x_t = np.zeros(22)
    y_t = np.zeros(22)
    for i in range(1,23):
        x_t[i-1] = temp_x.iloc[index]['x_{:0.0f}'.format(i)]
        y_t[i-1] = temp_x.iloc[index]['y_{:0.0f}'.format(i)]
    sender = temp_x.iloc[index]["sender"]
    print(temp_x.iloc[index]["time_start"])
    plt.plot(x_t[:11], y_t[:11], 'rs', x_t[11:], y_t[11:], 'g^', x_t[sender-1], y_t[sender-1], 'bo')
    for i in range(1, 23):
        plt.text(x_t[i-1], y_t[i-1],"{}".format(i))
    plt.show()











    """
    # Import prepared training set
    x_train_pairs = pd.read_csv('personal_set/training_x_pairs.csv', index_col=0, sep=',')
    y_train_pairs = pd.read_csv('personal_set/training_y_pairs.csv', index_col=0, sep=',')

    # Split the set:
    x_train, x_test, y_train, y_test = train_test_split(x_train_pairs, y_train_pairs, test_size=0.2)

    # Train the model:
    model.train(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy())

    # Import raw data never seen
    raw_x = pd.read_csv('input_training_set.csv', sep=',')
    raw_y = pd.read_csv('output_training_set.csv', sep=',')

    # keep only the 20% unused:
    start_index = int(raw_x.shape[0] * 0.8)
    tmp_x = np.copy(raw_x.to_numpy())[start_index:, :]
    tmp_y = np.copy(raw_y.to_numpy())[start_index:, :]
    raw_x = pd.DataFrame(data=tmp_x, columns=raw_x.columns)
    raw_y = pd.DataFrame(data=tmp_y, columns=raw_y.columns)

    # Make a prediction
    predictions = model.predict_pass(raw_x)

    # Compare with output

    true_class = 0
    for i in range(0, len(predictions)):
        print('{} - {}'.format(predictions[i], tmp_y[i]))
        if predictions[i] == tmp_y[i][0]:
            true_class += 1

    true_class /= tmp_y.shape[0]

    print('proportion bien classée: {}'.format(true_class))
    """
