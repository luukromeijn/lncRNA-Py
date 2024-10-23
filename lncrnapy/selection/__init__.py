'''A collection of feature selection and importance analysis methods.'''

from lncrnapy.selection.methods import (
    NoSelection, TTestSelection, RegressionSelection, ForestSelection, 
    RFESelection, PermutationSelection, MDSSelection
)

from lncrnapy.selection.importance_analysis import (
    feature_importance_analysis
)