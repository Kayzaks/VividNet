

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


            
    @staticmethod
    def mapDataOneWayRev(values : list, newMap : dict):
        # newMap        # Object - Index
        outputs : dict = dict()
        for obj, idx in newMap.items():
            outputs[obj] = values[idx]

        return outputs  # Object - Value
        

    @staticmethod
    def mapDataOneWayDict(values : dict, newMap : dict):
        # values        # Object - Value
        # newMap        # Index  - Object
        outputs = [0.0] * len(newMap)
        for idx, obj in newMap.items():
            if obj in values:
                outputs[idx] = values[obj]

        return outputs  # Values
        
        
    @staticmethod
    def mapDataOneWayDictRev(values : dict, newMap : dict):
        # values        # Object - Value
        # newMap        # Object - Index
        outputs = [0.0] * len(newMap)
        for obj, idx in newMap.items():
            if obj in values:
                outputs[idx] = values[obj]

        return outputs  # Values