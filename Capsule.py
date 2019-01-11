from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory

class Capsule:

    def __init__(self):
        self._attributes    : list          = list()  # Attribute
        self._memory        : CapsuleMemory = CapsuleMemory()


    def inheritAttributes(self, fromCapsules : list):
        xList : list = list() # Attributes
        for capsule in fromCapsules:
            for attribute in capsule._attributes:
                xList.append(attribute)
                # Make sure we don't have copies
                if attribute.getType() not in [x.getType() for x in self._attributes]:
                    newAttribute = attribute.getType().createAttribute()
                    newAttribute.setInherited()
                    self._attributes.append(newAttribute)

        self._memory.inferXAttributes(xList)
        self._memory.inferYAttributes(self._attributes)




