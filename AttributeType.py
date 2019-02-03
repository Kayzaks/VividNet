
from enum import Enum


class AttributeLexical(Enum):
    # Semantics
    Preposition = 1
    Adjective   = 2
    Verb        = 3

    # Visual
    Pixel       = 10

    # Non-Transmitted
    NonTransmit = 20


class AttributeType:

    def __init__(self, attributeName : str, attributeLexical : AttributeLexical):
        self._name          : str               = attributeName
        self._lexical       : AttributeLexical  = attributeLexical


    def getName(self):
        return self._name
