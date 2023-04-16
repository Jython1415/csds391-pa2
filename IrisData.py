# Class to import, parse, and access data from the Iris dataset
class IrisData:
    
    data = []
    
    def __init__(self):
        self.importData()
    
    def importData(self):
        with open("data/iris.data", "r") as dataFile:
            lines = dataFile.read().split("\n")
        self.data = [[float(x) for x in i.split(",")[:4]] for i in lines]
        for i, line in enumerate(lines):
            self.data[i].append(line.split(",")[4])

    def getData(self):
        return self.data
    
    def __str__(self):
        lines = [f"{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}" for line in self.getData()]
        return "\n".join(lines)