from AttributeType import AttributeType

class Attribute:

    def __init__(self, attributeType : AttributeType):
        self._type          : AttributeType = attributeType
        #self._value         : float         = 0.0
        self._isInherited   : bool          = False

    def getType(self):
        return self._type

    #def getValue(self):
    #    return self._value

    #def setValue(self, value : float):
    #    self._value = value

    def getName(self):
        return self._type.getName()

    def setInherited(self):
        self._isInherited = True

