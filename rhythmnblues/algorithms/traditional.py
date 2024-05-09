'''Pre-implemented coding/non-coding RNA classifiers.

Note that all classifiers that are based on algorithms presented in related
works should be considered as loose adaptations. We do not guarantee that our
implementations achieve the exact same performance as that of the original 
works.'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from rhythmnblues.data import Data
from rhythmnblues.features import *
from rhythmnblues.algorithms.algorithm import Algorithm


class CPAT(Algorithm): 
    '''Coding-Potential Assesment Tool (CPAT).
    
    References
    ----------
    CPAT: Wang et al. (2013) https://doi.org/10.1093/nar/gkt006'''

    def __init__(self, fickett_ref, hexamer_ref):
        '''Initializes CPAT algorithm based on a fickett reference file and 
        hexamer-score reference file.
        
        Arguments
        ---------
        `fickett_ref`: `str` | `Data`
            Path to Fickett reference file (fitted on a dataset) generated by a
            `FickettScore` object, or a `Data` object (training set) from
            which a new Fickett table should be calculated.
        `hexamer_ref`: `str` | `Data`
            Path to hexamer score bias reference file (fitted on a dataset)
            generated by a `KmerScore` object, or a `Data` object (training set)
            from which a new hexamer bias table should be calculated.'''
        
        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            FickettScore(fickett_ref),
            KmerScore(hexamer_ref, 6),
        )
        features = ['ORF length', 'ORF coverage', 'Fickett score', 
                    '6-mer score']
        model = make_pipeline(StandardScaler(), LogisticRegression())
        super().__init__(model, feature_extractors, features)


class CNCI(Algorithm):
    '''Coding Non-Coding Index (CNCI)
    
    References
    ----------
    CNCI: Sun et al. (2013) https://doi.org/10.1093/nar/gkt646''' 

    def __init__(self, ant_ref):
        '''Initializes CNCI algorithm based on ANT matrix reference file.
        
        Arguments
        ---------
        `ant_ref`: `str` | `Data`
            Path to ANT matrix reference file (fitted on a dataset) generated by
            a `MLCDS` object, or a `Data` object (training set) from which a new
           ANT matrix should be calculated.
        '''
        codon_extractor = KmerFreqs(3, apply_to='MLCDS1')
        feature_extractors = (
            MLCDS(ant_ref),
            MLCDSLength(),
            MLCDSLengthPercentage(),
            MLCDSScoreDistance(),
            codon_extractor,
        )
        # CNCI leaves out stop codons in codon bias
        features = ['MLCDS1 length', 'MLCDS1 score', 'MLCDS length-percentage', 
                    'MLCDS score-distance'] + [f'{kmer} (MLCDS1)' for kmer in 
                   codon_extractor.kmers if kmer not in ['TAA', 'TGA', 'TAG']]
        model = make_pipeline(StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)


class PLEK(Algorithm):
    '''Predictor of long non-coding RNAs and messenger RNAs based on an improved
    k-mer scheme (PLEK)
    
    References
    ----------
    PLEK: Li et al. (2014) https://doi.org/10.1186/1471-2105-15-311''' 

    def __init__(self):
        '''Initializes PLEK algorithm.'''
        feature_extractors = (
            [KmerFreqs(i) for i in range(1,6)] + 
            [KmerFreqs(i, PLEK=True) for i in range(1,6)]
        )
        features = ([f'{kmer} (PLEK)' for extractor in feature_extractors[:5] 
                     for kmer in extractor.kmers])
        model = make_pipeline(StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)

    def feature_extraction(self, data):
        '''Calls upon the object's feature extractors if a feature in the 
        `used_features` attribute is missing in `data`.
        
        Altered for PLEK algorithm to prevent requiring special PLEK nucleotide
        frequencies (will calculate them instead).'''

        # Check if k-mer PLEK frequencies are present
        if not data.check_columns(self.used_features, behaviour='bool'):
            kmers = ([f'{kmer}' for extractor in self.feature_extractors[:5] 
                     for kmer in extractor.kmers])
            # If not, check if normal k-mer frequencies are present
            if not data.check_columns(kmers, behaviour='bool'):
                # If not, calculate normal k-mer frequencies first...
                for extractor in self.feature_extractors[:5]:
                    data.calculate_feature(extractor)
            for extractor in self.feature_extractors[5:]:
                data.calculate_feature(extractor) # ... then scale like PLEK

        return data


class CNIT(Algorithm):
    '''Coding-Non-Coding Identifying Tool (CNIT)
    
    References
    ----------
    CNIT: Guo et al. (2019) https://doi.org/10.1093/nar/gkz400''' 

    def __init__(self, ant_ref):
        '''Initializes CNCI algorithm based on ANT matrix reference file.
        
        Arguments
        ---------
        `ant_ref`: `str` | `Data`
            Path to ANT matrix reference file (fitted on a dataset) generated by
            a `MLCDS` object, or a `Data` object (training set) from which a new
            ANT matrix should be calculated.'''
        codon_extractor = KmerFreqs(3, apply_to='MLCDS1')
        feature_extractors = (
            MLCDS(ant_ref),
            MLCDSLength(),
            MLCDSScoreStd(),
            MLCDSLengthStd(),
            codon_extractor,
        )
        features = (['MLCDS1 score', 'MLCDS score (std)', 'MLCDS length (std)'] 
                    + [f'{kmer} (MLCDS1)' for kmer in codon_extractor.kmers])
        model = make_pipeline(StandardScaler(),XGBClassifier(n_estimators=1000))
        super().__init__(model, feature_extractors, features)


class CPC(Algorithm):
    '''Adaptation of Coding Potential Calculator (CPC). This adaptation differs
    in two ways from the original: 1) we use ORF length instead of the log-odds
    score; 2) we do not make us of the ORF integrity feature, as all identified 
    ORFs will have a start and stop codon.
    
    References
    ----------
    CPC: Kong et al. (2007) https://doi.org/10.1093/nar/gkm391'''

    def __init__(self, database, **kwargs):
        '''Initializes CPC algorithm for given BLAST protein database.
        
        Arguments
        ---------
        `database`: `str`
            Path to local BLAST database or name of official BLAST database 
            (when running remotely).
        `**kwargs`
            Any keyword argument accepted by `BLASTXSearch` object.'''
        
        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            BLASTXSearch(database, **kwargs)
        )
        features = ['ORF length', 'ORF coverage', 'BLASTX hits', 
                    'BLASTX hit score', 'BLASTX frame score']
        model = make_pipeline(SimpleImputer(missing_values=np.nan), 
                              StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)


class CPC2(Algorithm):
    '''Adaptation of Coding Potential Calculator version 2 (CPC2). An important
    difference between this implementation and the original is that we do not
    make us of the ORF integrity feature, as all ORFs will have a start and stop
    codon.
    
    References
    ----------
    CPC2: Kang et al. (2017) https://doi.org/10.1093/nar/gkx428''' 

    def __init__(self, fickett_ref):
        '''Initializes CPC2 algorithm based Fickett reference file. 
        
        Arguments
        ---------
        `fickett_ref`: `str` | `Data`
            Path to Fickett reference file (fitted on a dataset) generated by a
            `FickettScore` object, or a `Data` object (training set) from
            which a new Fickett table should be calculated.'''
        feature_extractors = (
            FickettScore(fickett_ref),
            ORFCoordinates(),
            ORFProtein(),
            ORFLength(),
            ORFIsoelectric()
        )
        # CNCI leaves out stop codons in codon bias
        features = ['Fickett score', 'ORF length', 'ORF pI']
        model = make_pipeline(SimpleImputer(missing_values=np.nan), 
                              StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)


class FEELnc(Algorithm):
    '''FlExible Extraction of LncRNAs (FEELnc)
    
    References
    ----------
    FEELnc: Wucher et al. (2017) https://doi.org/10.1093/nar/gkw1306'''

    def __init__(self, kmer_refs):
        '''Initializes FEELnc algorithm for given `kmer_refs`, which is either a
        list of strings specifying the filepaths to k-mer bias reference files
        or a `Data` object from which the k-mer biases should be calculated.'''

        if type(kmer_refs) == Data:
            kmer_refs = [kmer_refs]*6
        elif type(kmer_refs) != list:
            raise TypeError()
        elif len(kmer_refs != 6): 
            raise ValueError()
        
        feature_extractors = (
            Length(),
            ORFCoordinates(relaxation=0),
            ORFCoordinates(relaxation=1),
            ORFCoordinates(relaxation=2),
            ORFCoordinates(relaxation=3),
            ORFCoordinates(relaxation=4),
            ORFLength(range(5)),
            ORFCoverage(range(5)),
            KmerScore(kmer_refs[0], k=1),
            KmerScore(kmer_refs[1], k=2),
            KmerScore(kmer_refs[2], k=3),
            KmerScore(kmer_refs[3], k=6),
            KmerScore(kmer_refs[4], k=9),
            KmerScore(kmer_refs[5], k=12),
        )

        features = ['length', 'ORF coverage', 'ORF1 coverage', 'ORF2 coverage',
                    'ORF3 coverage', 'ORF4 coverage', '1-mer score', 
                    '2-mer score', '3-mer score', '6-mer score', '9-mer score',
                    '12-mer score']
        model = make_pipeline(StandardScaler(), 
                              RandomForestClassifier(n_estimators=500)) 
        super().__init__(model, feature_extractors, features)


class iSeeRNA(Algorithm):
    '''Adaptation of the iSeeRNA algorithm. An important difference is that the
    conservation score feature is replaced by the number of BLASTX hits.
    
    References
    ----------
    iSeeRNA: Sun et al. (2013) https://doi.org/10.1186/1471-2164-14-S2-S7'''

    def __init__(self, database, **kwargs):
        '''Initializes iSeeRNA algorithm for given BLAST protein database.
        
        Arguments
        ---------
        `database`: `str`
            Path to local BLAST database or name of official BLAST database 
            (when running remotely).
        `**kwargs`
            Any keyword argument accepted by `BLASTXSearch` object.'''
        
        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            BLASTXSearch(database, **kwargs), # Replaces conservation score
            KmerFreqs(k=2),
            KmerFreqs(k=3)
        )
        features = ['ORF length', 'ORF coverage', 'BLASTX hits', 'GC', 'CT', 
                    'TAG', 'TGT', 'ACG', 'TCG']
        model = make_pipeline(StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)


class LncFinder(Algorithm): # NOTE not in unittests due to slow SSE features
    '''LncFinder algorithm. Utilizes log distance of (ORF) k-mer frequencies, as
    well as secondary structure elements and EIIP-derived physico-chemical
    features.
    
    References
    ----------
    LncFinder: Han et al. (2018) https://doi.org/10.1093/bib/bby065'''

    def __init__(self, orf_6mer_ref, acguD_4mer_ref, acguACGU_3mer_ref):
        '''Initializes LncFinder algorithm for given k-mer reference profiles or
        `Data` object to calculate these profiles for.
        
        Arguments
        ---------
        `orf_6mer_ref`: `str`|`Data`
            Path to ORF hexamer distance profiles file, or `Data` object for 
            calculating these profiles.
        `acguD_4mer_ref`: `str`|`Data`
            Path to acguD 4-mer distance profiles file, or `Data` object for 
            calculating these profiles.
        `acguACGU_3mer_ref`: `str`|`Data`
            Path to acgu-ACGU 3-mer distance profiles file, or `Data` object for 
            calculating these profiles.'''

        orf_6_mer = KmerDistance(orf_6mer_ref, 6, 'log', 'ORF', 3)
        acguD_4mer = KmerDistance(acguD_4mer_ref, 4, 'log', 'acguD', 1, 'ACGTD')
        acguACGU_3mer = KmerDistance(acguACGU_3mer_ref, 3, 'log', 'acgu-ACGU', 
                                     1, 'ACGTacgt')
        eiip_features = EIIPPhysicoChemical()
        
        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            orf_6_mer,
            SSE(),
            UPFrequency(),
            acguD_4mer, 
            acguACGU_3mer,
            eiip_features,
        )
        features = (['ORF length', 'ORF coverage', 'MFE', 'UP freq.'] + 
                    orf_6_mer.name + acguD_4mer.name + acguACGU_3mer.name +
                    eiip_features.name)
        model = make_pipeline(StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)


# NOTE Deep part still in progress
# NOTE Excluded from unittests due to BLASTX
class LncADeep(Algorithm): 
    '''Adaptation of the LncADeep algorithm, a feature-based classifier based on
    a Deep Belief Network. We replace HMMER with BLASTX, and implement the 
    full-length variant as explained in the paper.
    
    References
    ----------
    LncADeep: Yang et al. (2018) https://doi.org/10.1093/bioinformatics/bty428
    '''

    def __init__(self, fickett_ref, hexamer_ref, database, **kwargs):
        '''Initializes LncADeep algorithm for given reference files/data.
        
        Arguments
        ---------
        `fickett_ref`: `str` | `Data`
            Path to Fickett reference file (fitted on a dataset) generated by a
            `FickettScore` object, or a `Data` object (training set) from
            which a new Fickett table should be calculated.
        `hexamer_ref`: `str` | `Data`
            Path to hexamer score bias reference file (fitted on a dataset)
            generated by a `KmerScore` object, or a `Data` object (training set)
            from which a new hexamer bias table should be calculated.
        `database`: `str`
            Path to local BLAST database or name of official BLAST database 
            (when running remotely).
        `partial_length`: `bool`
            If `True`, will use the feature etup tailored for partial-length 
            sequences as specified in the paper.
        `**kwargs`
            Any keyword argument accepted by `BLASTXSearch` object.'''
        
        orf_kmer_freqs = KmerFreqs(2, 'ORF')
        edp_orf_kmer_freqs = EntropyDensityProfile(orf_kmer_freqs.name)

        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            orf_kmer_freqs,
            edp_orf_kmer_freqs,
            KmerScore(hexamer_ref, 6, apply_to='ORF'),
            UTRLength(),
            UTRCoverage(),
            GCContent(apply_to='UTR5'),
            GCContent(apply_to='UTR3'),
            FickettScore(fickett_ref),
            BLASTXSearch(database, **kwargs)
        )

        features = (['ORF length', 'ORF coverage'] + edp_orf_kmer_freqs.name +
                    ['6-mer score (ORF)', 'UTR5 coverage', 'UTR3 coverage', 
                     'GC content (UTR5)', 'GC content (UTR3)', 
                     'Fickett score', 'BLASTX hits', 'BLASTX hit score'])
        model = make_pipeline(SimpleImputer(missing_values=np.nan), # TODO:
                              StandardScaler(), SVC()) # Replace with deep-learn
        super().__init__(model, feature_extractors, features)


class PLncPro(Algorithm): # NOTE no unittests because of BLAST dependency
    '''Plant Long Non-Coding RNA Prediction by Random fOrest
    
    References
    ----------
    PLncPro: Singh et al. (2017) https://doi.org/10.1093/nar/gkx866'''

    def __init__(self, database, **kwargs):
        '''Initializes PLncPro algorithm for given BLAST protein database.
        
        Arguments
        ---------
        `database`: `str`
            Path to local BLAST database or name of official BLAST database 
            (when running remotely).
        `**kwargs`
            Any keyword argument accepted by `BLASTXSearch` object.'''
        
        trimers = KmerFreqs(3)

        feature_extractors = (
            Length(),
            BLASTXSearch(database, **kwargs),
            trimers,
        )
        features = ['length', 'BLASTX hits', 'BLASTX S-score', 
                    'BLASTX bit score', 'BLASTX frame entropy'] + trimers.name
        model = make_pipeline(SimpleImputer(missing_values=np.nan), 
                              StandardScaler(), 
                              RandomForestClassifier(n_estimators=1000))
        super().__init__(model, feature_extractors, features)


class CPPred(Algorithm): 
    '''Adaptation of Coding Potential Prediction (CPPred). We replace C and T 
    of CTD features with mono- and dimer frequencies, respectively.
    
    References
    ----------
    CPPred: Tong et al. (2019) https://doi.org/10.1093/nar/gkz087'''

    def __init__(self, fickett_ref, hexamer_ref):
        '''Initializes CPPred algorithm based on a fickett reference file and 
        hexamer-score reference file.
        
        Arguments
        ---------
        `fickett_ref`: `str` | `Data`
            Path to Fickett reference file (fitted on a dataset) generated by a
            `FickettScore` object, or a `Data` object (training set) from
            which a new Fickett table should be calculated.
        `hexamer_ref`: `str` | `Data`
            Path to hexamer score bias reference file (fitted on a dataset)
            generated by a `KmerScore` object, or a `Data` object (training set)
            from which a new hexamer bias table should be calculated.'''
        
        monomers = KmerFreqs(1) # Replaces C (composition) of CTD
        dimers = KmerFreqs(2) # Replaces T (transition) of CTD
        distribution = SequenceDistribution()

        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            FickettScore(fickett_ref),
            KmerScore(hexamer_ref, 6),
            ORFProtein(),
            ORFProteinAnalysis(),
            monomers,
            dimers,
            distribution,
        )
        features = (['ORF length', 'ORF coverage', 'Fickett score', 
                     '6-mer score', 'ORF pI', 'ORF gravy', 'ORF instability'] + 
                    monomers.name + dimers.name + distribution.name)
        model = make_pipeline(SimpleImputer(missing_values=np.nan), 
                              StandardScaler(), SVC())
        super().__init__(model, feature_extractors, features)


class DeepCPP(Algorithm): # NOTE: currently missing its deep component
    '''Deep neural network for coding potential prediction (DeepCPP).
    
    References
    ----------
    DeepCPP: Zhang et al. (2020) https://doi.org/10.1093/bib/bbaa039'''

    def __init__(self, fickett_ref, hexamer_ref, zhang_ref):
        '''Initializes `DeepCPP` algorithm based on a fickett reference file and 
        hexamer-score reference file.
        
        Arguments
        ---------
        `fickett_ref`: `str` | `Data`
            Path to Fickett reference file (fitted on a dataset) generated by a
            `FickettScore` object, or a `Data` object (training set) from
            which a new Fickett table should be calculated.
        `hexamer_ref`: `str` | `Data`
            Path to hexamer score bias reference file (fitted on a dataset)
            generated by a `KmerScore` object, or a `Data` object (training set)
            from which a new hexamer bias table should be calculated.
        `zhang_ref`: `str` | `Data`
            Path to Zhang score bias reference file (fitted on a dataset)
            generated by a `ZhangScore` object, or a `Data` object (training 
            set) from which a new bias reference should be calculated.'''
        
        monomer = KmerFreqs(1)
        codons = KmerFreqs(3, apply_to='ORF')
        bigap = KmerFreqs(2, gap_length=2, gap_pos=2)
        trigap = KmerFreqs(2, gap_length=3, gap_pos=2)

        feature_extractors = (
            Length(),
            ORFCoordinates(),
            ORFLength(),
            ORFCoverage(),
            FickettScore(fickett_ref),
            KmerScore(hexamer_ref, 6),
            ZhangScore(zhang_ref),
            monomer,
            codons,
            bigap,
            trigap
        )
        features = (['ORF length', 'ORF coverage', 'Fickett score', 
                    '6-mer score', 'Zhang score'] + monomer.name + codons.name +
                    bigap.name + trigap.name)
        model = make_pipeline(SimpleImputer(), StandardScaler(), 
                              RandomForestClassifier(n_estimators=1000))
        super().__init__(model, feature_extractors, features)