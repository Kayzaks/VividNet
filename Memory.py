

class Memory:
    
    def nextBatch(self, batchSize : int, inputMap : dict, outputMap : dict):
        # inputMap  : dict   # Object - List of Indices
        # outputMap : dict   # Object - List of Indices
        yData = [[]] * batchSize
        xData = [[]] * batchSize

        return (xData, yData)  # List of X Values, List of Y Values