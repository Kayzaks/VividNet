
from AttributeType import AttributeType
from AttributeType import AttributeLexical
from Attribute import Attribute

class AttributePool:

    def __init__(self):
        self._pool : dict = dict() # Name - AttributeType

    def newType(self, attributeName : str, attributeLexical : AttributeLexical):
        if attributeName in self._pool:
            print("AttributeType named " + attributeName + " already exists!")
            return False
        else:
            self._pool[attributeName] = AttributeType(attributeName, attributeLexical)
            return True

    def createAttribute(self, attributeName : str = None):
        if attributeName in self._pool:
            return self._pool[attributeName].createAttribute()
        else:
            print("No AttributeType named " + attributeName + " exists in the pool")
            return None
            