import math

class DataEntry:
    
    def __init__(self, params, label):
        
        if type(params) == list:
            self.params = tuple(params)
        elif type(params) == tuple:
            self.params = params
        else:
            raise Exception(f"\"params\" parameter must be either a tuple or a list. {type(params)} is not allowed.")
        
        self.label = label
        
    def getParams(self):
        return self.params
    
    def getLabel(self):
        return self.label
    
    def getEntry(self):
        return (self.params, self.label)
    
class Dataset:
    
    def __init__(self, listOfEntries):
        self.data = listOfEntries
    
    def getPoints(self):
        return [entry.getParams() for entry in self.data]
   
class Cluster:
    
    def __init__(self, centroid = None):
        self.centroid = centroid
        self.points = []
    
    def setCentroid(self, newCentroid):
        self.centroid = newCentroid
    
    def getCentroid(self):
        return self.centroid
    
    def addPoint(self, point):
        self.points.append(point)
    
    def clearPoints(self):
        self.points.clear()
        
class KMeansClustering:
    
    data:Dataset = None
    clusters:list[Cluster] = []
    
    # rawData must be an array with float parameters and one label entry (str or int)
    def __init__(self, rawData, k = 3, maxItr = 1000):
        self.k = k
        self.maxItr = maxItr
        self.rawData = rawData.copy
        
    def prepareData(self):
        
        paramIndices = []
        labelIndex = -1   # Stores where the label is in the raw data
        data = []
        
        for index, entry in enumerate(self.rawData[0]):
            if type(entry) == float:
                self.paramIndices.append(index)
            elif type(entry) == str or type(entry) == int:
                if labelIndex == -1:
                    self.labelIndex = index
                else:
                    raise Exception(f"rawData has more than one label: {labelIndex} and {index}")
        
        for row in self.rawData:
            data.append(DataEntry([row[i] for i in paramIndices], row[labelIndex]))
    
        self.data = Dataset(data)
        
    def objectiveFunction(self):
        total = 0 # Holds the sum of the objective function
        
        # Iterate over all the points
        for point in self.data.getPoints():
            # Add the distance to the closest centroid
            total += min([math.dist(cluster.getCentroid(), point) ** 2 for cluster in self.clusters])
        
        return total
        
        