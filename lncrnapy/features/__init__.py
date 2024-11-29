'''Contains feature extractor classes that can calculate several features of RNA
sequences, such as Most-Like Coding Sequences and nucleotide frequencies. 

Every feature extractor class contains:
* A `name` attribute of type `str`, indicating what name a `Data` column for
this feature will have.
* A `calculate` method with a `Data` object as argument, returning a list or
array of the same length as the `Data` object.'''

from lncrnapy.features.blast import (BLASTXSearch, BLASTXBinary, BLASTNSearch,
                                     BLASTNCoverage)
from lncrnapy.features.fickett import FickettScore
from lncrnapy.features.general import (
    Length, Complexity, Entropy, EntropyDensityProfile, GCContent, 
    StdStopCodons, SequenceDistribution, Quality
)
from lncrnapy.features.kmer import (
    KmerDistance, KmerFreqs, KmerScore
)
from lncrnapy.features.mlcds import (
    MLCDS, MLCDSLength, MLCDSLengthPercentage, MLCDSLengthStd, 
    MLCDSScoreDistance, MLCDSScoreStd
)
from lncrnapy.features.orf import (
    ORFAminoAcidFreqs, ORFCoordinates, ORFCoverage, ORFIsoelectric, ORFLength, 
    ORFProtein, ORFProteinAnalysis, UTRLength, UTRCoverage
)
try: # Allows ViennaRNA package to be optional
    from lncrnapy.features.sse import SSE, UPFrequency
except ModuleNotFoundError:
    pass
from lncrnapy.features.tokenizers import KmerTokenizer, BytePairEncoding
from lncrnapy.features.zhang import ZhangScore
from lncrnapy.features.eiip import EIIPPhysicoChemical
from lncrnapy.features.standardizer import Standardizer
from lncrnapy.features.mlm_accuracy import MLMAccuracy