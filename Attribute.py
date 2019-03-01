from AttributeType import AttributeType

class Attribute:

    def __init__(self, attributeType : AttributeType):
        self._type          : AttributeType = attributeType
        self._isInherited   : bool          = False

    def getType(self):
        return self._type

    def getName(self):
        return self._type.getName()

    def setInherited(self):
        self._isInherited = True

