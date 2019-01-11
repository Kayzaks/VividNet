from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory
from CapsuleRoute import CapsuleRoute

class Capsule:

    def __init__(self):
        self._attributes    : list          = list()  # Attribute
        self._routes        : list          = list()  # Route


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


    def getAttributes(self):
        return self._attributes

