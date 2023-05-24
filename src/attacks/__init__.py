from enum import Enum


class TOGAttacks(Enum):
    # TOGAttacks.fabrication, TOGAttacks.vanishing
    vanishing = 0
    untargeted = 1
    fabrication = 2
    mislabellingML = 3
    mislabellingLL = 4
    universal = 5
