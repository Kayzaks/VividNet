

from Observation import Observation
from CapsuleNetwork import CapsuleNetwork
from RelationTriplet import RelationTriplet
from AttributePool import AttributePool
        
class PrimitivesPhysics:

    def __init__(self, attributePool : AttributePool):
        self._attributePool = attributePool


    # -------------------- User Defined
    
    def generateRelation(self):
        triplet = [0.0]
        effect  = [0.0]
        return triplet, effect

    def generateInteraction(self):
        aggregate  = [0.0]
        attributes = [0.0]
        return aggregate, attributes

    # --------------------


    def render(self, capsNet : CapsuleNetwork, observationFrames : list, width : int, height : int):
        # observationFrames     # List of Dictionaries {capsule, List of Observations}

        frames = []
        for observationFrame in observationFrames:

            imageReal, semantics, texts = capsNet.generateImage(width, height, observationFrame, False)

            drawPixels = [0.0] * (width * height * 3)
            
            for yy in range(height):
                for xx in range(width):
                    drawPixels[(yy * width + xx) * 3] = imageReal[(yy * width + xx) * 4]
                    drawPixels[(yy * width + xx) * 3 + 1] = imageReal[(yy * width + xx) * 4]
                    drawPixels[(yy * width + xx) * 3 + 2] = imageReal[(yy * width + xx) * 4]

            frames.append(drawPixels)

        return frames   # Renderable frames for the UI
