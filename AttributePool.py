
from AttributeType import AttributeType
from AttributeType import AttributeLexical
from Attribute import Attribute


class AttributePool:

    def __init__(self):
        self._pool : dict = dict() # AttributeType - AttributeType


    def createType(self, attributeName : str, attributeLexical : AttributeLexical):
        if self.getTypeByName(attributeName) is not None:
            print("AttributeType named " + attributeName + " already exists!")
            return False
        else:
            self._pool[AttributeType(attributeName, attributeLexical)] = []
            return True


    def createAttribute(self, attributeName : str = None):
        currentType = self.getTypeByName(attributeName)
        if currentType is not None:
            self._pool[currentType].append(Attribute(currentType))
            return self._pool[currentType][-1]
        else:
            print("No AttributeType named " + attributeName + " exists in the pool")
            return None
            

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


    def getTypeByName(self, attributeName : str):
        return next( (x for x in self._pool if x.getName() == attributeName), None)