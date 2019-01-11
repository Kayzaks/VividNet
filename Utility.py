

class Utility:

    @staticmethod
    def mapData(values : list, originalMap : dict, newMap : dict):
        # originalMap   # Index  - Object
        # newMap        # Object - Index
        outputs = [0.0] * len(newMap)
        for idx, val in enumerate(values):
            outputs[newMap[originalMap[idx]]] = val 

        return outputs # Values

            
    @staticmethod
    def mapDataOneWay(values : list, newMap : dict):
        # newMap        # Index - Object
        outputs : dict = dict()
        for idx, val in enumerate(values):
            outputs[newMap[idx]] = val 

        return outputs  # Object - Value