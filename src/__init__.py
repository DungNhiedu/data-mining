# =============================================================================
# DO AN: Ap dung Cay quyet dinh va Naive Bayes phan tich xu huong ket hon
#        cua gioi tre Viet Nam (18-35)
# =============================================================================
# File: __init__.py
# Mo ta: Package initialization
# =============================================================================

from .data_loader import (
    create_combined_dataset,
    load_marriage_data,
    load_tfr_data,
    load_population_data,
    get_summary_by_region,
    get_summary_by_year
)
from .preprocessing import (
    load_data,
    load_real_data,
    split_data, 
    get_tree_preprocessor, 
    get_nb_preprocessor,
    get_feature_info,
    FEATURES_REAL,
    TARGET_REAL,
    FEATURES,
    TARGET
)
from .decision_tree_model import DecisionTreeModel, train_and_compare_trees
from .naive_bayes_model import NaiveBayesModel, compare_nb_models

__all__ = [
    # Data loading
    'create_combined_dataset',
    'load_marriage_data',
    'load_tfr_data',
    'load_population_data',
    'get_summary_by_region',
    'get_summary_by_year',
    # Preprocessing
    'load_data',
    'load_real_data',
    'split_data',
    'get_tree_preprocessor',
    'get_nb_preprocessor',
    'get_feature_info',
    'FEATURES_REAL',
    'TARGET_REAL',
    'FEATURES',
    'TARGET',
    # Models
    'DecisionTreeModel',
    'train_and_compare_trees',
    'NaiveBayesModel',
    'compare_nb_models'
]
