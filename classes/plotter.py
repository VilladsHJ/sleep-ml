
class Plotter():
    def __init__(self, app):
        self.app = app
        print(self.__class__.__name__+" ready! Requested by "+self.app.__class__.__name__)
    
    def plot(self, segmentData):
        print("Plotting... "+str(segmentData))


if __name__ == "__main__":
    print("Plotter main")
