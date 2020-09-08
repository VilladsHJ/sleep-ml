from dataLoader import DataLoader
from segmenter import Segmenter
from plotter import Plotter

class App():
    def __init__(self):
        self.d = DataLoader(self, dataPath)
        self.s = Segmenter(self)
        self.p = Plotter(self)

    def run(self):
        self.s.segment()
        self.p.plot(self.s.segmentData)


if __name__ == "__main__":
    a = App()
    a.run()
