from kNN.kernels.Base import Base


class Quartic(Base):
    def __init__(self, metric):
        super().__init__(metric)

    def function(self, u):
        return 15 * ((1 - u ** 2) ** 2) / 16
