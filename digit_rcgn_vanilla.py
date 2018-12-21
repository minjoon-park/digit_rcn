import random
import time

import numpy as np
import scipy.special as sp

INPUT_DIM = 28
HIDDEN_LAYER_DIM = 10
OUTPUT_DIM = 10

TRAINING_PREFIX = 'train'
TEST_PREFIX = 't10k'

'''
Vanilla, single layer implementation
Overshoot protection

mem: ~485M
performance:
    default
        accuracy: 84.810000 after 1000 runs in 409.756796s
    learn rate: 2, reg:
        1       89.1
        0.9     accuracy: 89.260000 after 1000 runs in 399.315042s
        0.85    accuracy: 89.160000 after 1000 runs in 400.967612s
        0.8     accuracy: 89.940000 after 1000 runs in 403.872361s
        0.75    accuracy: 89.470000 after 1000 runs in 401.305660s
        0.7     accuracy: 89.940000 after 1000 runs in 403.872361s
        0.6     accuracy: 90.100000 after 1000 runs in 405.122192s
        0.55    accuracy: 89.790000 after 1000 runs in 403.176063s
        0.5     89.3
        0.1     86
'''


class DigitRecognition(object):

    # synapse_ih[HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM][INPUT_DIM * INPUT_DIM]
    # synapse_ho[OUTPUT_DIM][HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM]

    def __init__(self,
                 working_dir='C:\\Users\\Minjoon\\Documents\\',
                 learn_rate=1.,
                 regularization_strength=3.,
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
        self.synapse_ih = np.random.randn(
            INPUT_DIM * INPUT_DIM, HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM
        ) / np.sqrt(OUTPUT_DIM)
        self.synapse_ih_b = np.zeros(
            shape=(HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM)
        )
        self.synapse_ho = np.random.randn(
            HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM, OUTPUT_DIM
        ) / np.sqrt(OUTPUT_DIM)
        self.synapse_ho_b = np.zeros(shape=(OUTPUT_DIM))

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
                images.append(image)
                if not image_file.peek(1):
                    break
        setattr(self, prefix + '_images', np.array(images))

    def _get_batch(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(random.randrange(self.num_train))
        return set(batch)

    def feed_forward(self, prefix, image_no):
        hidden = getattr(self, prefix + '_images')[image_no].dot(
            self.synapse_ih
        ) + self.synapse_ih_b
        hidden = sp.expit(hidden)

        output = hidden.dot(self.synapse_ho) + self.synapse_ho_b
        output = np.exp(output)
        output = output / output.sum()

        return hidden, output

    def train(self):
        step_ih = np.zeros(shape=(
            INPUT_DIM * INPUT_DIM, HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM
        ))
        step_ih_b = np.zeros(shape=(HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM))
        step_ho = np.zeros(shape=(
            HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM, OUTPUT_DIM
        ))
        step_ho_b = np.zeros(shape=(OUTPUT_DIM))
        prev_loss = self.loss
        self.loss = 0.

        for image_no in self._get_batch():
            hidden, output = self.feed_forward(TRAINING_PREFIX, image_no)
            self.loss -= self.train_labels[image_no].dot(np.log(output))

            tmp_ho = self.train_labels[image_no] - output
            step_ho_b -= tmp_ho
            step_ho -= hidden.reshape(
                HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM, 1
            ).dot(tmp_ho.reshape(1, OUTPUT_DIM))

            tmp_ih = tmp_ho.dot(self.synapse_ho.T) * hidden * (1. - hidden)
            step_ih_b -= tmp_ih
            step_ih -= self.train_images[image_no].reshape(
                INPUT_DIM * INPUT_DIM, 1
            ).dot(tmp_ih.reshape(1, HIDDEN_LAYER_DIM * HIDDEN_LAYER_DIM))

        self.loss += self.regularization_strength / 2 * (
            np.sum(np.square(self.synapse_ih)) +
            np.sum(np.square(self.synapse_ho))
        )
        self.loss /= self.batch_size

        step_ih_b += self.regularization_strength * self.synapse_ih_b
        step_ih_b = step_ih_b / self.batch_size
        self.synapse_ih_b -= self.learn_rate * step_ih_b
        step_ih += self.regularization_strength * self.synapse_ih
        step_ih = step_ih / self.batch_size
        self.synapse_ih -= self.learn_rate * step_ih

        step_ho_b += self.regularization_strength * self.synapse_ho_b
        step_ho_b = step_ho_b / self.batch_size
        self.synapse_ho_b -= self.learn_rate * step_ho_b
        step_ho += self.regularization_strength * self.synapse_ho
        step_ho = step_ho / self.batch_size
        self.synapse_ho -= self.learn_rate * step_ho

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

        accuracy = 100. * correct / self.num_t10k
        print('\taccuracy: %f' % accuracy)
        return accuracy
