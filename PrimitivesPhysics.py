

from Observation import Observation
from CapsuleNetwork import CapsuleNetwork
from RelationTriplet import RelationTriplet
from AttributePool import AttributePool
        
class PrimitivesPhysics:

    def __init__(self, attributePool : AttributePool):
        self._attributePool = attributePool

        self.init()


    # -------------------- User Defined

    def init(self):
        return


    def generateInteractionSequence(self, capsNet : CapsuleNetwork, width : int, height : int, folder : str, idname : str):
        # Generate Images in the folder with name id + "." + sequence_index + file_format

        return
    
    
    def generateRelation(self):
        # Triplet Format:
        # Sender   -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic
        # Receiver -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic 
        # Relation -- Distance | Degrees-Of-Freedom | Sender Normal | Receiver Normal

        # Effect Format:
        # Acceleration Vector | Angle Acceleration Vector

        triplet = [0.0]
        effect  = [0.0]
        return triplet, effect


    def generateInteraction(self):
        # Aggregate Format:
        # Receiver -- Attributes | Symbol | Velocities | Static/Dynamic | Rigid/Elastic
        # Effects  -- Summed Effect Acceleration Vector | Summed Effect Angle Acceleration Vector
        # External -- External Acceleration Vector | External Angle Acceleration Vector

        # Attributes Format:
        # Receiver -- Attributes | Accelerations

        aggregate  = [0.0]
        attributes = [0.0]
        return aggregate, attributes

    # --------------------
