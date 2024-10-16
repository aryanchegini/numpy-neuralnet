import numpy as np
from utils import shuffle_dataset

train_data = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)
test_data = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)


def save_weights_and_biases(weights, biases, filename):
    weights_and_biases = {'weights': weights, 'biases': biases}
    np.save(filename, weights_and_biases)

def load_weights_and_biases(filename):
    data = np.load(filename, allow_pickle=True).item()
    return data['weights'], data['biases']

class Network:
    def __init__(self, layers, weights=None, biases=None):
        self.layers = layers
        self.weights = weights if weights is not None else [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = biases if biases is not None else [np.random.randn(y, 1) for y in layers[1:]]
        self.activation_func = Network.sigmoid
        self.activation_func_deriv = Network.sigmoid_derivative
        
    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.activation_func(np.dot(w, x) + b)
        return x

    def one_hot_encode(self, y):
        one_hot_encoded = np.zeros((self.layers[-1], 1), dtype=int)  # Make this a column vector
        one_hot_encoded[y, 0] = 1
        return one_hot_encoded

    def backpropagation(self, x, y):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b
            a = self.activation_func(z)
            zs.append(z)
            activations.append(a)
            x = a

        # Calculate the error in the last layer (output layer)
        delta = (activations[-1] - self.one_hot_encode(y)) * self.activation_func_deriv(zs[-1])
        deltas = [delta]

        # Find the errors in each layer going backwards from the last layer
        for l in range(2, len(self.layers)):
            delta = np.dot(np.transpose(self.weights[-l + 1]), deltas[-l + 1]) * self.activation_func_deriv(zs[-l])
            deltas.insert(0, delta)

        # Use the errors for each layer to compute the partial derivatives of the cost function w.r.t. the weights and biases 
        delta_b = deltas
        delta_w = [np.dot(deltas[-l], np.transpose(activations[-l - 1])) for l in range(len(self.layers) - 1, 0, -1)]
        return delta_w, delta_b

    def make_prediction(self, x):
        return np.argmax(self.feed_forward(x))

    def train(self, train_ds, mini_batch_size, epochs=1000, learning_rate=1, test_ds=None):
        """Implements Stochastic Gradient Descent using mini-batches"""
        for epoch in range(epochs+1):
            num_correct = 0

            # Shuffle dataset by combining, shuffling, and then splitting again

            np.random.shuffle(train_ds)
            x_train = train_ds[:, 1:]
            y_train = train_ds[:, 0]

            x_test = test_ds[:, 1:]
            y_test = test_ds[:, 0]

            # Split the train dataset into batches of a specified number
            batches = Network.get_mini_batches(x_train, y_train, mini_batch_size)

            for batch in batches:
                avg_del_w = [np.zeros(w.shape) for w in self.weights]
                avg_del_b = [np.zeros(b.shape) for b in self.biases]

                for x, y in zip(batch[0], batch[1]):
                    # Change the shape of the input vector from (54,) to (54, 1)
                    x = np.array(x).reshape(-1, 1)

                    num_correct = num_correct + 1 if (self.make_prediction(x) == y) else num_correct

                    del_w, del_b = self.backpropagation(x, y)
                    avg_del_w = [w1 + w2 for (w1, w2) in zip(avg_del_w, del_w)]
                    avg_del_b = [b1 + b2 for (b1, b2) in zip(avg_del_b, del_b)]

                # Compute the average partial derivatives for each mini-batch and update the weights and biases
                avg_del_w = [w * (1 / len(batch[0])) for w in avg_del_w]
                avg_del_b = [b * (1 / len(batch[0])) for b in avg_del_b]

                # learning rate decreases exponentionly to allow the weights and biases to converge
                self.weights = [w1 - (learning_rate*(np.exp((-2.3026 / epochs)*epoch))) * w2 for (w1, w2) in zip(self.weights, avg_del_w)]
                self.biases = [b1 - (learning_rate*(np.exp((-2.3026 / epochs)*epoch))) * b2 for (b1, b2) in zip(self.biases, avg_del_b)]

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}: Train accuracy: {num_correct / len(y_train)}")
                if test_ds is not None:  # Validate on test data after each epoch
                    correct_count = self.evaluate(x_test, y_test)
                    print(f"Test accuracy: {correct_count / len(y_test)}")
        return self.weights, self.biases

    def evaluate(self, x_test, y_test):
        return sum(1 for x, y in zip(x_test, y_test) if self.make_prediction(np.array(x).reshape(-1, 1)) == y)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return Network.sigmoid(x) * (1 - Network.sigmoid(x))

    @staticmethod
    def get_mini_batches(x_train, y_train, mini_batch_size):
        return [(x_train, y_train) for x_train, y_train in
                zip(np.array_split(x_train, int(np.ceil(x_train.shape[0] / mini_batch_size))),
                    np.array_split(y_train, int(np.ceil(y_train.shape[0] / mini_batch_size))))]
        
    def predict(self, data):
        return np.array([self.make_prediction(x.reshape(-1, 1)) for x in data])
    

def create_classifier():
    # weights, biases = load_weights_and_biases("saved_weights_and_biases.npy")
    classifier = Network([54, 30, 2])
    # for training:
    opt_w, opt_b = classifier.train(train_data, 30, 200, 1, test_ds=test_data)
    return classifier

if __name__ == "__main__":
    net1 = create_classifier()
    
