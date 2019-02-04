from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory
from CapsuleRoute import CapsuleRoute

class Capsule:

    def __init__(self, name : str):
        self._name          : str           = name    # Capsule Name / Symbol
        self._attributes    : list          = list()  # Attribute
        self._routes        : list          = list()  # Route


    def addNewRoute(self, fromCapsules : list):
        numRoutes = len(self._routes)
        newRoute = CapsuleRoute(self, self._name + "-R-" + str(numRoutes), fromCapsules)
        self._routes.append(newRoute)


    def inheritAttributes(self, fromCapsules : list):
        for route in self._routes:
            for capsule in route.getFromCapsules():
                for attribute in capsule.getAttributes():
                    # Make sure we don't have copies
                    if attribute.getType() not in [x.getType() for x in self._attributes]:
                        newAttribute = attribute.getType().createAttribute()
                        newAttribute.setInherited()
                        self._attributes.append(newAttribute)

            route.resizeInternals()


    def createAttribute(self, name : str, attributePool : AttributePool):
        newAttribute = attributePool.createAttribute(name)
        if newAttribute is not None:
            self._attributes.append(newAttribute)

        for route in self._routes:
            route.resizeInternals()
        

    def getAttributeByName(self, name : str):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                return attr
        
        return None


    def getAttributes(self):
        return self._attributes


    def getMappedAttributes(self, outputMap : dict):
        # outputMap     # Index - Attribute
        outputList = []
        for key, value in sorted(outputMap.items()):
            outputList.append(value.getValue())

        return outputList


    def getAttributeValue(self, name : str):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                return attr.getValue()
        
        return 0.0

    def setAttributeValue(self, name : str, value : float):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                attr.setValue(value)

