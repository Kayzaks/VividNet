

class HyperParameters:
    # Capsules
    SimilarObservationsCutOff = 0.9 
    PrimitiveProbabilityCutOff = 0.75 
    SemanticProbabilityCutOff = 0.85 

    # Primitive Capsule Agreement
    PrimAgreementWidth = 0.0
    PrimAgreementFallOff = 1.0 # 0.4

    # Semantic Capsule Agreement
    SemAgreementWidth = 0.05
    SemAgreementFallOff = 0.15

    # Symmetry
    SymmetryCutOff = 0.95

    # Learning
    AdamLearningRate = 0.0001