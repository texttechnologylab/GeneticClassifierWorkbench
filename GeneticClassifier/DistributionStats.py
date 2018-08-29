import math

class DistributionStats(object):

    def __init__(self, distribution):
        self.distribution = distribution
        self.min = 0
        self.max = 0
        self.avg = 0
        self.stddev = 0
        for i in range(len(distribution)):
            self.avg = self.avg + distribution[i]
            if i == 0:
                self.min = distribution[i]
                self.max = distribution[i]
            else:
                if distribution[i] < self.min:
                    self.min = distribution[i]
                if distribution[i] > self.max:
                    self.max = distribution[i]
        self.avg = self.avg / len(distribution)
        for i in range(len(distribution)):
            self.stddev = self.stddev + (distribution[i] - self.avg) * (distribution[i] - self.avg)
        self.stddev = math.sqrt(self.stddev / len(distribution))
