

class HyperParameters:
    # Capsules
    SimilarObservationsCutOff = 0.9 
    PrimitiveProbabilityCutOff = 0.75 
    SemanticProbabilityCutOff = 0.75 
    SizeCutoff = 0.08

    # Primitive Capsule Agreement
    PrimAgreementWidth = 0.0
    PrimAgreementFallOff = 1.0 

    # Semantic Capsule Agreement
    SemAgreementWidth = 0.0
    SemAgreementFallOff = 0.40

    # Symmetry
    SymmetryCutOff = 0.99

    # Learning
    AdamLearningRate = 0.0001

    # Physics 
    MaximumAttributeCount = 10
    MaximumSymbolCount = 10
    Dimensions = 2
    DegreesOfFreedom = Dimensions * 3 - 3
    TimeStep = 1.0 / 24.0
    AccelerationScale = 10.0 
    VelocityCutoff = 0.1 # 0.2
    DistanceCutoff = 0.01
    ContinuityCutoff = TimeStep * 1.5