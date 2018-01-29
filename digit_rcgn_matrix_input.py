import random
import time

from math import exp
from math import log
import numpy as np
import scipy.special as sp

INPUT_DIM = 28
HIDDEN_LAYER_DIM = 10
OUTPUT_DIM = 10

TRAINING_PREFIX = 'train'
TEST_PREFIX = 't10k'

'''
Feed input in a matrix form
Overshoot protection

mem: ~487M
performance:
    default
        accuracy: 65.330000 after 1000 runs in 185.902133s
    learn rate: 2, reg:
    	1.5    accuracy: 63.560000 after 1000 runs in 187.068504s
    	1.2    accuracy: 63.670000 after 1000 runs in 188.775752s
    	1.1    accuracy: 65.350000 after 1000 runs in 188.080622s
        1      accuracy: 70.230000 after 1000 runs in 210.820340s
        0.9    accuracy: 65.640000 after 1000 runs in 188.318634s
        0.8    accuracy: 68.560000 after 1000 runs in 188.724834s
        0.7    accuracy: 35.490000 after 1000 runs in 206.475967s
        0.6    accuracy: 11.350000 after 1000 runs in 209.100841s
        0.5    accuracy: 65.220000 after 1000 runs in 203.225898s
'''


class DigitRecognition(object):

    def __init__(self,
                 working_dir='C:\\Users\\Minjoon\\Documents\\',
                 learn_rate=1.,
                 regularization_strength=1.,
                 batch_size=100):
        self.working_dir = working_dir
        self.learn_rate = learn_rate
        self.regularization_strength = regularization_strength
        self.batch_size = batch_size
        self.train_labels = None
        self.train_images = None
        self.num_train = 0
        self.t10k_labels = None
        self.t10k_images = None
        self.num_t10k = 0
        self.loss = 100.
        self.synapse_ih_f = np.random.randn(
            HIDDEN_LAYER_DIM, INPUT_DIM
        ) / np.sqrt(OUTPUT_DIM)
        self.synapse_ih_b = np.random.randn(
            INPUT_DIM, HIDDEN_LAYER_DIM
        ) / np.sqrt(OUTPUT_DIM)
        self.synapse_ih_bias = np.zeros(
            shape=(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)
        )
        self.synapse_ho_f = np.random.randn(
            OUTPUT_DIM, HIDDEN_LAYER_DIM
        ) / np.sqrt(OUTPUT_DIM)
        self.synapse_ho_b = np.random.randn(
            HIDDEN_LAYER_DIM, 1
        ) / np.sqrt(OUTPUT_DIM)
        self.synapse_ho_bias = np.zeros(shape=(OUTPUT_DIM, 1))

        self.load_data(TRAINING_PREFIX)
        self.load_data(TEST_PREFIX)

    def run(self):
        acc, cnt = 0., 0
        start = time.time()
        while acc < 99. and cnt < 1000:
            cnt += 1
            self.train()
            acc = self.test()
        print('accuracy: %f after %d runs in %fs' % (
            acc, cnt, time.time() - start
        ))

    def load_data(self, prefix):
        data_cnt = 0
        labels = []
        with open(
            self.working_dir + prefix + "-labels.idx1-ubyte", 'rb'
        ) as label_file:
            label_file.read(8)
            while True:
                label = label_file.read(1)
                if not label:
                    break
                data_cnt += 1
                label = label[0]
                label_one_hot = np.zeros(shape=(OUTPUT_DIM, 1))
                label_one_hot[label, 0] = 1
                labels.append(label_one_hot)
        setattr(self, 'num_' + prefix, data_cnt)
        setattr(self, prefix + '_labels', np.array(labels))

        images = []
        with open(
            self.working_dir + "\\" + prefix + "-images.idx3-ubyte", 'rb'
        ) as image_file:
            image_file.read(16)
            while True:
                image = []
                for i in range(INPUT_DIM * INPUT_DIM):
                    image.append(image_file.read(1)[0] / 255.)
                images.append(np.array(image).reshape(INPUT_DIM, INPUT_DIM))
                if not image_file.peek(1):
                    break
        setattr(self, prefix + '_images', np.array(images))

    def _get_batch(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(random.randrange(self.num_train))
        return set(batch)

    def feed_forward(self, prefix, image_no):
        hidden = self.synapse_ih_f\
        	.dot(getattr(self, prefix + '_images')[image_no])\
        	.dot(self.synapse_ih_b) + self.synapse_ih_bias
        hidden = sp.expit(hidden)

        output = self.synapse_ho_f.dot(hidden).dot(self.synapse_ho_b) + \
        	self.synapse_ho_bias
        output = np.exp(output)
        output = output / output.sum()

        return hidden, output

    def train(self):
        step_ih_f = np.zeros(shape=(HIDDEN_LAYER_DIM, INPUT_DIM))
        step_ih_b = np.zeros(shape=(INPUT_DIM, HIDDEN_LAYER_DIM))
        step_ih_bias = np.zeros(shape=(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM))
        step_ho_f = np.zeros(shape=(OUTPUT_DIM, HIDDEN_LAYER_DIM))
        step_ho_b = np.zeros(shape=(HIDDEN_LAYER_DIM, 1))
        step_ho_bias = np.zeros(shape=(OUTPUT_DIM, 1))
        prev_loss = self.loss
        self.loss = 0.

        for image_no in self._get_batch():
            hidden, output = self.feed_forward(TRAINING_PREFIX, image_no)
            self.loss -= self.train_labels[image_no].T.dot(np.log(output))[0, 0]

            tmp_ho = self.train_labels[image_no] - output
            step_ho_bias -= tmp_ho
            step_ho_f -= tmp_ho.dot(hidden.dot(self.synapse_ho_b).T)
            step_ho_b -= self.synapse_ho_f.dot(hidden).T.dot(tmp_ho)

            tmp_ih = self.synapse_ho_f.T.dot(tmp_ho)\
            	.dot(self.synapse_ho_b.T) * hidden * (1. - hidden)
            step_ih_bias -= tmp_ih
            step_ih_f -= tmp_ih\
            	.dot(self.train_images[image_no].dot(self.synapse_ih_b).T)
            step_ih_b -= self.synapse_ih_f.dot(self.train_images[image_no]).T\
            	.dot(tmp_ih)

        self.loss += self.regularization_strength / 2 * (
            np.sum(np.square(self.synapse_ih_f)) +
            	np.sum(np.square(self.synapse_ih_b)) +
                np.sum(np.square(self.synapse_ho_f)) +
                np.sum(np.square(self.synapse_ho_b))
        )
        self.loss /= self.batch_size

        step_ih_bias += self.regularization_strength * self.synapse_ih_bias
        step_ih_bias = step_ih_bias / self.batch_size
        self.synapse_ih_bias -= self.learn_rate * step_ih_bias
        step_ih_f += self.regularization_strength * self.synapse_ih_f
        step_ih_f = step_ih_f / self.batch_size
        self.synapse_ih_f -= self.learn_rate * step_ih_f
        step_ih_b += self.regularization_strength * self.synapse_ih_b
        step_ih_b = step_ih_b / self.batch_size
        self.synapse_ih_b -= self.learn_rate * step_ih_b

        step_ho_bias += self.regularization_strength * self.synapse_ho_bias
        step_ho_bias = step_ho_bias / self.batch_size
        self.synapse_ho_bias -= self.learn_rate * step_ho_bias
        step_ho_f += self.regularization_strength * self.synapse_ho_f
        step_ho_f = step_ho_f / self.batch_size
        self.synapse_ho_f -= self.learn_rate * step_ho_f
        step_ho_b += self.regularization_strength * self.synapse_ho_b
        step_ho_b = step_ho_b / self.batch_size
        self.synapse_ho_b -= self.learn_rate * step_ho_b

        print('loss: %f, change: %f' % (
            self.loss, 100 * (prev_loss - self.loss) / prev_loss
        ))

        if prev_loss < self.loss and self.learn_rate > .01:
            self.learn_rate *= .9

    def test(self):
        correct = 0

        for image_no in range(self.num_t10k):
            _, output = self.feed_forward(TEST_PREFIX, image_no)
            if np.argmax(output) == np.argmax(self.t10k_labels[image_no]):
                correct += 1

        accuracy = 100. * correct / self.num_t10k;
        print('\taccuracy: %f' % accuracy)
        return accuracy
