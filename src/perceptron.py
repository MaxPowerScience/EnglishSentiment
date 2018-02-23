import os
import tflearn
import datetime
import sklearn
import numpy as np

def create_perceptron(max_sequence_length, dict_size):

    net = tflearn.input_data([None, max_sequence_length])
    net = tflearn.embedding(net, input_dim=dict_size+1, output_dim=128)
    net = tflearn.fully_connected(net, 100, activation='relu')
    net = tflearn.fully_connected(net, 20, activation='relu')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1,
                             loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='../tensorboard/tensorboard_fully')

def train_network(trainX, trainY, model):
    batch_size = 24

    save_folder = '../models/perceptron/'
    save_name = 'perceptron_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.tfl'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path = save_folder + save_name

    #n_epoch = 100
    #for i in range(0,n_epoch):
    trainX, trainY = tflearn.data_utils.shuffle(trainX, trainY)
    model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=batch_size, n_epoch=50)

    model.save(save_path)
    return model

def test_network(testX, testY, model):

    # Transform acutal class labels (1 for positiv and 0 for negative)
    actualClasses = []
    for test in testY:
        actualClasses.append(1) if test[0] == 1 else actualClasses.append(0)

    # Predict class for test data

    pred = np.array(model.predict(testX))
    predictions = (np.array(model.predict(testX))[:,1] >= 0.5).astype(np.int_)

    # Print classification report and confusion matrix
    #print(sklearn.metrics.classification_report(actualClasses, predictions))
    #print(sklearn.metrics.confusion_matrix(actualClasses, predictions))

    # Accuracy of test prediction
    test_accuracy = np.mean(predictions == testY[:,1], axis=0)
    print("Test accuracy: ", test_accuracy)

def main():
    create_perceptron()

if __name__ == "__main__":
    main()