import numpy as np
import logging


class Main(object):
    def __init__(self):
        self.N = 1
        self.x = 0.8
        self.d = 0.72
        self.eta = 0.3

        self.W1 = [0.3, -0.3]
        self.b1 = [0, 0]
        self.W2 = [-0.1, 0.1]
        self.b2 = [0]

        self.net_j = None
        self.z_j = None
        self.net_k = None
        self.z_k = None
        self.y = None
        self.delta_1j = None
        self.delta_2k = None
        self._w1 = None
        self._w2 = None

    def feed_forward(self, _input):
        """

        :param x:
        :param w1:
        :param w2:
        :return: net_j, z_j, net_k, y
        """
        self.net_j = np.multiply(self.W1, _input) + self.b1
        self.z_j = np.tanh(self.net_j)

        self.net_k = np.sum(np.multiply(self.W2, self.z_j)) + self.b2
        self.z_k = self.net_k
        self.y = self.z_k
        return self.y
        pass

    def backward_propagation(self, pred):
        self.delta_2k = (self.d - pred) * 1

        self.delta_1j = np.multiply(
            (1 - np.power(np.tanh(self.net_j), 2)),
            np.multiply(self.W2, self.delta_2k))

        return self.delta_2k, self.delta_1j
        pass

    def update(self):
        u_w2 = - (self.eta * np.multiply(self.delta_2k, self.z_j))
        u_w1 = - (self.eta * np.multiply(self.delta_1j, self.x))

        u_b2 = - (self.eta * self.delta_2k)
        u_b1 = - (self.eta * self.delta_1j)

        # renew
        _w1 = self.W1 - u_w1  # new weight_ji
        _w2 = self.W2 - u_w2  # new weight_kj
        _b1 = self.b1 - u_b1  # new bias_ji
        _b2 = self.b2 - u_b2  # new bias_kj

        # delta para.
        d_W1 = _w1 - self.W1
        d_W2 = _w2 - self.W2
        d_b1 = _b1 - self.b1
        d_b2 = _b2 - self.b2

        # update
        self.W1 = _w1
        self.W2 = _w2
        self.b1 = _b1
        self.b2 = _b2

        return d_W1, d_W2, d_b1, d_b2
        pass


if __name__ == '__main__':

    _ = Main()
    iteration = 302
    print("================================== t: ", "0", " ==================================")
    print("W1/b1: ", _.W1, "/", _.b1)
    print("W2/b2: ", _.W2, "/", _.b2)

    for itr in range(1, iteration):
        pred = _.feed_forward(_input=_.x)
        delta2_k, delta_1j = _.backward_propagation(pred=pred)
        d_W1, d_W2, d_b1, d_b2 = _.update()
        print("Z: ", _.z_j)
        print("pred: ", pred)
        print("delta1, delta2: ", delta_1j, delta2_k)
        print("d_W1/d_b1: ", d_W1, "/", d_b1)
        print("d_W2/d_b2: ", d_W2, "/", d_b2)
        print("================================== t: ", itr, " ==================================")
        print("W1/b1: ", _.W1, "/", _.b1)
        print("W2/b2: ", _.W2, "/", _.b2)
        # logging.warning("================= t: {0} =================".format(itr))
        # logging.warning("W1/b1: {0}/{1} | W2/b2: {2}/{3} | Z: {4} | y: {5}".format(
        #      _.W1, _.b1, _.W2, _.b2, _.z_j, pred))


