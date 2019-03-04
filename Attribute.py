from AttributeType import AttributeType
from AttributeType import AttributeLexical

class Attribute:

    def __init__(self, attributeType : AttributeType):
        self._type          : AttributeType = attributeType
        self._isInherited   : bool          = False

    def getType(self):
        return self._type

    def getName(self):
        return self._type.getName()

    def isInheritable(self):
        if self._type._lexical != AttributeLexical.NonTransmit and self._type._lexical != AttributeLexical.Pixel:
            return True
        else:
            return False

    def setInherited(self):
        self._isInherited = True

