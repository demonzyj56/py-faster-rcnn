import numpy as np
import caffe


class simplified_frcnn(object):

    def __init__(self, solver_prototxt):
        self.solver = caffe.SGDSolver(solver_prototxt)
