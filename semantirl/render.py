import numpy as np
from collections import namedtuple

from semantirl.utils.colors import print_color

Cell = namedtuple('Cell', ['text', 'back'])

class Canvas(object):
    def __init__(self, shape):
        self.canvas = np.array(np.zeros(shape), dtype=np.object_)
        for idx, val in np.ndenumerate(self.canvas):
            self.canvas[idx] = Cell(' ', None)
        self.shape = shape

    def set_fore(self, x, y, text, bcolor=None):
        old = self.canvas[x,y]
        self.canvas[x,y] = Cell(text, old.back)

    def set_back(self, x, y, bcolor):
        old = self.canvas[x,y]
        self.canvas[x,y] = Cell(old.text, bcolor)

    def display(self):
        canvas = np.fliplr(self.canvas).T
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cell = canvas[i,j]
                print_color(cell.text, back=cell.back)
            print '\n',

