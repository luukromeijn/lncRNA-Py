'''A collection of feature selection and importance analysis methods.'''

from rhythmnblues.selection.methods import (
    NoSelection, TTestSelection, RegressionSelection, ForestSelection, 
    RFESelection, PermutationSelection, MDSSelection
)

from rhythmnblues.selection.importance_analysis import (
    feature_importance_analysis
)