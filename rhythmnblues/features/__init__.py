'''Contains feature extractor classes that can calculate several features of RNA
sequences, such as Most-Like Coding Sequences and nucleotide frequencies. 

Every feature extractor class contains:
* A `name` attribute of type `str`, indicating what name a `Data` column for
this feature will have.
* A `calculate` method with a `Data` object as argument, returning a list or
array of the same length as the `Data` object.'''

from rhythmnblues.features.blast import BLASTXSearch, BLASTXBinary
from rhythmnblues.features.fickett import FickettTestcode
from rhythmnblues.features.general import Length, Complexity, FeatureEntropy
from rhythmnblues.features.kmer import (
    KmerDistance, KmerFreqs, KmerFreqsPLEK, KmerScore
)
from rhythmnblues.features.mlcds import (
    MLCDS, MLCDSKmerFreqs, MLCDSLength, MLCDSLengthPercentage, MLCDSLengthStd, 
    MLCDSScoreDistance, MLCDSScoreStd
)
from rhythmnblues.features.orf import (
    ORFAminoAcidFreqs, ORFCoordinates, ORFCoverage, ORFIsoelectric, ORFLength, 
    ORFProtein, ORFProteinAnalysis
)
try: # Allows ViennaRNA package to be optional
    from rhythmnblues.features.sse import SSE, UPFrequency
except ModuleNotFoundError:
    pass