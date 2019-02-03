from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory
from CapsuleRoute import CapsuleRoute

class Capsule:

    def __init__(self):
        self._attributes    : list          = list()  # Attribute
        self._routes        : list          = list()  # Route


    # TODO: New Route
    # 1. Inherit Attributes
    # 2. Create CapsuleRoute

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


    def getAttributes(self):
        return self._attributes

    def getAttributeValue(self, name : str):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                return attr.getValue()
        
        return 0.0

    def setAttributeValue(self, name : str, value : float):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                attr.setValue(value)

