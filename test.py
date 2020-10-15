class A:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)

    def __key(self):
        return self.x, self.y

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, A):
            return self.__key() == other.__key()
        return NotImplemented

mydict = {}

mydict[(1,2)] = 'a'
mydict[(1,3)] = 'b'
mydict[(1,2)] = 'c'

breakpoint()