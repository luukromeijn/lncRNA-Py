'''Pre-implemented coding/non-coding RNA classifiers and a base `Algorithm` 
class for creating new classification algorithms.

Note that all classifiers that are based on algorithms presented in related
works should be considered as loose adaptations. We do not guarantee that our
implementations achieve the exact same performance as that of the original 
works.'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from rhythmnblues.data import Data
from rhythmnblues.feature_extraction import *


class Algorithm:
    '''Base class for custom coding/non-coding RNA classifiers.
    
    Attributes
    ----------
    `feature_extractors`: `list`
        List of feature extractors that are applied to the data if a feature in 
        `used_features` is missing in the input.
    `used_features`: `list[str]`
        List of feature names (data columns) that serve as explanatory variables
        for the model.
    `model`:
        Model with `.fit` and `.classify` method, base model of classifier.'''

    def __init__(self, feature_extractors, used_features, model):
        '''Initializes `Algorithm` object.
        
        Arguments
        ---------
        `feature_extractors`: `list`
            List of feature extractors that will be applied to the data if a 
            feature in `used_features` is missing in the input.
        `used_features`: `list[str]`
            List of feature names (data columns) that will serve as explanatory
             variables for the model.
        `model`:
            Model with `.fit` and `.classify` method, base model of classifier.
        '''

        self.feature_extractors = feature_extractors
        self.used_features = used_features
        self.model = model

    def fit(self, data):
        '''Fits model on `data`, extracting features first if necessary. Will
        only fit on features as specified in the `used_features` attribute.'''
        data = self.feature_extraction(data)
        y = data.df['label'].replace({'pcrna':0, 'ncrna':1})
        y = y.infer_objects(copy=False) # Downcasting from str to int
        self.model.fit(data.df[self.used_features], y)

    def predict(self, data):
        '''Classifies `data`, extracting features first if necessary. Will
        only use features as specified in the `used_features` attribute.'''
        self.feature_extraction(data)
        y = self.model.predict(data.df[self.used_features])
        y = np.vectorize({0:'pcrna', 1:'ncrna'}.get)(y)
        return y 

    def feature_extraction(self, data):
        '''Calls upon the object's feature extractors if a feature in the 
        `used_features` attribute is missing in `data`.'''
        if not data.check_columns(self.used_features, behaviour='bool'):
            for extractor in self.feature_extractors:
                data.calculate_feature(extractor)
        return data


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
            `FickettTestcode` object, or a `Data` object (training set) from
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
            FickettTestcode(fickett_ref),
            KmerScore(hexamer_ref, 6),
        )
        features = ['ORF length', 'ORF coverage', 'Fickett TESTCODE', 
                    '6-mer score']
        model = make_pipeline(StandardScaler(), LogisticRegression())
        super().__init__(feature_extractors, features, model)


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
        codon_extractor = MLCDSKmerFreqs(3)
        feature_extractors = (
            MLCDS(ant_ref),
            MLCDSLength(),
            MLCDSLengthPercentage(),
            MLCDSScoreDistance(),
            codon_extractor,
        )
        # CNCI leaves out stop codons in codon bias
        features = ['MLCDS1 length', 'MLCDS1 score', 'MLCDS length-percentage', 
                    'MLCDS score-distance'] + [f'{kmer} (MLCDS)' for kmer in 
                   codon_extractor.kmers if kmer not in ['TAA', 'TGA', 'TAG']]
        model = make_pipeline(StandardScaler(), SVC())
        super().__init__(feature_extractors, features, model)


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
            [KmerFreqsPLEK(i) for i in range(1,6)]
        )
        features = ([f'{kmer} (PLEK)' for extractor in feature_extractors[:5] 
                     for kmer in extractor.kmers])
        model = make_pipeline(StandardScaler(), SVC())
        super().__init__(feature_extractors, features, model)

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
        codon_extractor = MLCDSKmerFreqs(3)
        feature_extractors = (
            MLCDS(ant_ref),
            MLCDSLength(),
            MLCDSScoreStd(),
            MLCDSLengthStd(),
            codon_extractor,
        )
        features = (['MLCDS1 score', 'MLCDS score (std)', 'MLCDS length (std)'] 
                    + [f'{kmer} (MLCDS)' for kmer in codon_extractor.kmers])
        model = make_pipeline(StandardScaler(),XGBClassifier(n_estimators=1000))
        super().__init__(feature_extractors, features, model)


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
        super().__init__(feature_extractors, features, model)


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
            `FickettTestcode` object, or a `Data` object (training set) from
            which a new Fickett table should be calculated.'''
        feature_extractors = (
            FickettTestcode(fickett_ref),
            ORFCoordinates(),
            ORFProtein(),
            ORFLength(),
            ORFIsoelectric()
        )
        # CNCI leaves out stop codons in codon bias
        features = ['Fickett TESTCODE', 'ORF length', 'ORF pI']
        model = make_pipeline(SimpleImputer(missing_values=np.nan), 
                              StandardScaler(), SVC())
        super().__init__(feature_extractors, features, model)


class FEELnc(Algorithm):
    '''TODO
    
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
        super().__init__(feature_extractors, features, model)