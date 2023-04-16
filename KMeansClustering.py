import math
import random
from statistics import mean
from IrisData import IrisData
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
        
        if type(centroid) == tuple:
            self.centroid = centroid
        elif type(centroid) == list:
            self.centroid = tuple(centroid)
        else:
            raise Exception(f"\"centroid\" parameter must be either a tuple or a list. {type(centroid)} is not allowed.")
        
        self.points = []
    
    def setCentroid(self, newCentroid):
        self.centroid = newCentroid
    
    def getCentroid(self):
        return self.centroid
    
    def addPoint(self, point):
        self.points.append(point)
    
    def clearPoints(self):
        self.points.clear()
        
    def getPoints(self):
        return self.points
    
    def findCentroid(self):
        self.setCentroid(tuple([mean(column) for column in zip(*self.getPoints())]))
        
        
class KMeansClustering:
    
    data:Dataset = None         # Stores the data (points and labels)
    clusters:list[Cluster] = [] # Stores the k clusters
    
    # Output data
    objFunc = []
    clusterMaps = []
    
    # rawData must be an array with float parameters and one label entry (str or int)
    def __init__(self, rawData:IrisData, k = 3, maxItr = 1000):
        self.k = k
        self.maxItr = maxItr
        self.rawData = rawData.getData().copy()
        
    def prepareData(self):
        
        paramIndices = []
        labelIndex = -1   # Stores where the label is in the raw data
        data = []
        
        for index, entry in enumerate(self.rawData[0]):
            if type(entry) == float:
                paramIndices.append(index)
            elif type(entry) == str or type(entry) == int:
                if labelIndex == -1:
                    labelIndex = index
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
    
    def initialize(self):
        self.prepareData()
        self.clusters.clear()
        
        # Create k random clusters
        for _ in range(self.k):
            randomPosition = []
            for i in range(len(self.data.getPoints()[0])):
                allPoints = [point[i] for point in self.data.getPoints()]
                randomPosition.append(random.uniform(min(allPoints), max(allPoints)))
            self.clusters.append(Cluster(randomPosition))
    
    def iterate(self):
        # Assign points to clusters
        for cluster in self.clusters:
            cluster.clearPoints()
        
        for point in self.data.getPoints():
            distances = [math.dist(cluster.getCentroid(), point) ** 2 for cluster in self.clusters]
            self.clusters[distances.index(min(distances))].addPoint(point)
        
        # Find new centroid for each cluster
        for cluster in self.clusters:
            cluster.findCentroid()
    
    def run(self):
        self.initialize()
        
        for _ in range(self.maxItr):
            self.iterate()
            
            # Record data
            self.objFunc.append(self.objectiveFunction())
            # self.clusterMaps.append(self.plotClusters())
            
            # Check end condition
            if len(self.objFunc) >= 2 and self.objFunc[-1] == self.objFunc[-2]:
                break
        
    def plotClusters(self):
        # Get the colors
        colors = sorted(mcolors.TABLEAU_COLORS, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
        colors = list(reversed(colors))
        colors = [colors[i*3 % len(colors)] for i in range(len(colors))]
        colors = colors[:len(self.clusters)]
        
        # Plot the values
        fig, ax = plt.subplots()
        for i, cluster in enumerate(self.clusters):
            ax.scatter([point[2] for point in cluster.getPoints()], [point[3] for point in cluster.getPoints()], s=10, color=colors[i])
            ax.scatter(cluster.getCentroid()[2], cluster.getCentroid()[3], s=30, color='r', alpha=.5)
        
        ax.set_xlabel('Sepal Length')
        ax.set_ylabel('Petal Length')
        ax.set_title('Petal Length vs. Sepal Length with Colored Clusters')
        
        return fig