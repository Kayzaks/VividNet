
from AttributeType import AttributeType
from AttributeType import AttributeLexical
from Attribute import Attribute


class AttributePool:

    def __init__(self):
        self._pool  : dict = dict() # AttributeType - AttributeList
        self._names : dict = dict() # AttributeName - AttributeType


    def createType(self, attributeName : str, attributeLexical : AttributeLexical):
        if attributeName in self._names:
            print("AttributeType named " + attributeName + " already exists")
            return False
        else:
            currentType = AttributeType(attributeName, attributeLexical)
            self._pool[currentType] = []
            self._names[attributeName] = currentType
            return True


    def createAttribute(self, attributeName : str = None):
        if attributeName not in self._names:
            print("No AttributeType named " + attributeName + " exists in the pool")
            return None
        else:
            currentType = self._names[attributeName]
            self._pool[currentType].append(Attribute(currentType))
            return self._pool[currentType][-1]
            

    def destroyAttribute(self, attribute: Attribute):
        try: 
            if attribute.getType() in self._pool:
                self._pool[attribute.getType()].remove(attribute)
                return True
            else:
                print("Trying to remove attribute with an unknown type:")
                print(attribute.getType())
        except ValueError:
            print("Trying to remove attribute that shouldn't exist:")
            print(attribute)

        return False