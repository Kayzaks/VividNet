
from AttributeType import AttributeType
from AttributeType import AttributeLexical
from Attribute import Attribute


class AttributePool:

    def __init__(self):
        self._pool  : dict = dict() # AttributeType - AttributeList
        self._names : dict = dict() # AttributeName - AttributeType
        self._order : list = list() # List of AttributeNames


    def createType(self, attributeName : str, attributeLexical : AttributeLexical):
        if attributeName in self._names:
            return False
        else:
            currentType = AttributeType(attributeName, attributeLexical)
            self._pool[currentType] = []
            self._names[attributeName] = currentType
            self._order.append(attributeName)
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


    def getAttributeOrder(self, attribute : Attribute):
        if attribute.getName() in self._order:
            return self._order.index(attribute.getName())
        else:
            return -1


    def getAttributeOrderByName(self, attributeName : str):
        if attributeName in self._order:
            return self._order.index(attributeName)
        else:
            return -1


    def getAttributeNameByOrder(self, index : int):
        if index < len(self._order):
            return self._order[index]
        else:
            return ""

            
