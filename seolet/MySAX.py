import numpy as np


class MySAX():

    def __init__(self, n_bins, strategy="custom", compress=1, symbols = None):
        self.thr = None
        self.n_bins = n_bins
        self.strategy = strategy
        self.compress = compress
        if symbols is None:
            self.symbols = [x for x in "abcdefghijklmnopqrstuvwxyz"+"0123456789"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        else:
            self.symbols = symbols

        if n_bins > len(self.symbols):
            raise ValueError(f"n_bins must be at most {len(self.symbols)}")

    def fit(self, X=None):
        self.thr = np.linspace(-np.pi, np.pi, self.n_bins + 1)
        return self

    def transform(self, X):
        offset = np.pi / self.n_bins

        lista = []
        for i in range(0, len(X), self.compress):
            el = np.mean(X[i:i + self.compress])

            for minimo, massimo, lettera in zip(self.thr[:-1], self.thr[1:], iter(self.symbols)):
                if el - offset > -np.pi:
                    if minimo < el-offset <= massimo:
                        lista.append(lettera)
                        break
                else:
                    if minimo < el-offset+2*np.pi <= massimo:
                        lista.append(lettera)
                        break

        return np.array(lista).reshape((1, -1))