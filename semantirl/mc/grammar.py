import random

class GrammarNode(object):
    def __init__(self):
        self.__children = []

    def execute(self, s0, mc):
        for child in self.__children:
            s0 = child.execute(s0, mc)

    def describe(self):
        if random.random()>0.8 or self.isleaf():
            return random.choice(self._describe())
        return '. '.join([c.describe() for c in self.__children])

    def _describe(self):
        raise NotImplementedError()

    def isleaf(self):
        return len(self.__children) == 0

    def add_child(self, node):
        self.__children.append(node)


class MoveRel(GrammarNode):
    def __init__(self, x, z):
        super(MoveRel, self).__init__()
        self.x = x
        self.z = z

    def execute(self, s0, mc):
        raise NotImplementedError()

    def _describe(self):
        return ['move %d, %d' % (self.x, self.z),
                'move %d forward and %d right' % (self.x, self.z)]


class Place(GrammarNode):
    def __init__(self, block_type):
        super(Place, self).__init__()
        self.block_type = block_type

    def execute(self, s0, mc):
        raise NotImplementedError()

    def _describe(self):
        return ['place %s' % self.block_type,
                'place a %s block' % self.block_type,
                'put a %s block' % self.block_type,
                'put %s' % self.block_type]


class Line(GrammarNode):
    def __init__(self, material, start, end):
        super(Line, self).__init__()
        self.material = material
        self.start = start
        self.end = end

        self.add_child(MoveRel(start[0], start[1]))
        self.add_child(Place(material))

    def _describe(self):
        return ["draw a %s line from %s to %s" % (self.material, self.start, self.end)]

if __name__ == "__main__":
    l = Line('stone', (1,2), (3,4))
    print l.describe()

