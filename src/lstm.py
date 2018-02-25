import tflearn
import tensorflow as tf
import datetime
import numpy as np
import random

from random import randint

def create_lstm(max_sequence_length, dict_size, word_vectors):
    net = tflearn.input_data([None, max_sequence_length])

    #vocab_dim = 50
    #n_symbols = dict_size
    #embedding_weights = np.zeros((n_symbols, vocab_dim))
    #for word, index in index_dict.items():
    #    embedding_weights[index, :] = word_vectors[word]

    # define inputs here
    #embedding_layer = Embedding(output_dim=vocab_dim, input_dim=n_symbols, trainable=True)
    #embedding_layer.build((None,))  # if you don't do this, the next step won't work
    #embedding_layer.set_weights([embedding_weights])

    emb = tflearn.embedding(net, input_dim=dict_size, output_dim=50, trainable=True, name='EmbeddingLayer')
    net = tflearn.lstm(emb, 64, dropout=0.75)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='../tensorboard/tensorboard_lstm')

    embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
    model.set_weights(embeddingWeights, word_vectors)
    print(model.get_weights(emb.W))

    return model

def create_lstm_with_tensorflow(word_vectors, trainY):

        with tf.Session() as sess:
            print('bla')

        ids = np.load('../resources/idsMatrix.npy')

        # Construct LSTM

        maxSeqLength = 250
        batchSize = 24
        lstmUnits = 64
        numClasses = 2
        iterations = 100000
        numLayers = 1
        learningRate = 0.001
        keep_prob = 0.75
        numDimensions = 300

        tf.reset_default_graph()

        labels = tf.placeholder(tf.float32, [batchSize, numClasses])

        # shape input data (24, 250)
        input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

        # shape data (24, 250, 300)
        data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)

        # shape data (24, 250, 50)
        data = tf.nn.embedding_lookup(word_vectors, input_data)

        print(data.shape)

        lstmCell = tf.contrib.rnn.MultiRNNCell([get_a_cell(lstmUnits, keep_prob) for _ in range(numLayers)])

        # shape value (24, 250, 10)
        value, blub = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        print(accuracy)
        merged = tf.summary.merge_all()

        logdir = "../tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Train LSTM

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = getTrainBatch(batchSize, maxSeqLength, ids, trainY);

            # print(nextBatch.tolist())
            # print(nextBatch.shape)

            # print(nextBatchLabels)
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            # Write summary to Tensorboard
            if (i % 50 == 0):
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                print(summary, i)
                writer.add_summary(summary, i)
        writer.close()

def getTrainBatch(batchSize, maxSeqLength, ids, trainY):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(0,11499)
            #num = randint(1, 6999)
            labels.append([1,0])
        else:
            num = randint(13500,24999)
            #num = randint(14000, 20999)
            labels.append([0,1])
        arr[i] = ids[num]
    return arr, labels

def get_a_cell(lstm_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop