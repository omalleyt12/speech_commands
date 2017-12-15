class RunningAverage:
    def __init__(self):
        self.total = 0
        self.items = 0

    def add(self,val):
        self.total += val
        self.items += 1

    def calculate(self):
        if self.items == 0: return 0
        return self.total / self.items

    # def __add__(self,val):
    #     n = RunningAverage()
    #     if type(val) == type(self):
    #         n.total = self.total + val.total
    #         n.items = self.items + val.items
    #         return n
    #     else:
    #         n.total = self.total + val
    #         n.items = self.items + 1
    #     return n

    def __float__(self):
        return self.calculate()

    def __int__(self):
        return int(self.calculate())

    def __str__(self):
        return str(self.calculate())

    def __repr__(self):
        return self.__str__()
