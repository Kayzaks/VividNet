
from Attribute import Attribute
from enum import Enum


class AttributeLexical(Enum):
    Preposition = 1
    Adjective   = 2
    Verb        = 3


class AttributeType:

    def __init__(self, attributeName : str, attributeLexical : AttributeLexical):
        self._name          : str               = attributeName
        self._references    : list              = list()            # Attribute
        self._lexical       : AttributeLexical  = attributeLexical


    def getName(self):
        return self._name


    def createAttribute(self):
        outAttribute : Attribute = Attribute(self)

        self._references.append(outAttribute)

        return outAttribute


    def destroyAttribute(self, attribute: Attribute):
        try:
            self._references.remove(attribute)
            return True
        except ValueError:
            print("Trying to remove attribute that shouldn't exist.")
            print(attribute)

        return False