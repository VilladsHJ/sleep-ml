
class Segmenter():
    def __init__(self, app):
        self.app = app
        print(self.__class__.__name__+" ready! Requested by "+self.app.__class__.__name__)
        self.segmentData = [0]*10
        
    def segment(self):
        print(self.segmentData)
        self.segmentData = [1 for i in self.segmentData]
        print(self.segmentData)


if __name__ == "__main__":
    print("Segmenter main")
