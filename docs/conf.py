# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'lncRNA-Py'
copyright = '2024, Luuk Romeijn'
author = 'Luuk Romeijn'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "luukromeijn", # Username
    "github_repo": "rhythmnblues", # Repo name
    "github_version": "master", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}

rst_prolog = '''

.. |bert_archs| image:: ../experiments/figures/hyppar_tuning/bert_architectures.png
    :alt: BERT architectures
    :width: 660

.. |warmup| image:: ../experiments/figures/hyppar_tuning/warmup.png
    :width: 300
    :alt: Learning Curve warmup steps

.. |bpe_bs| image:: ../experiments/figures/hyppar_tuning/mlm_batch_sizes_bpe.png
    :alt: MLM BPE batch sizes
    :width: 300

.. |cse_bs| image:: ../experiments/figures/hyppar_tuning/mlm_batch_sizes_cse.png
    :alt: MLM CSE batch sizes
    :width: 300

.. |cls_lr_bpe| image:: ../experiments/figures/hyppar_tuning/finetune_lr_bpe.png
    :width: 220

.. |cls_lr_new| image:: ../experiments/figures/hyppar_tuning/finetune_lr-bs_bpe-cse.png
    :width: 220

.. |cls_lr_scratch| image:: ../experiments/figures/hyppar_tuning/finetune_lr_scratch_cse.png
    :width: 220

.. |cls_wd_bpe| image:: ../experiments/figures/hyppar_tuning/finetune_wd_bpe.png
    :width: 300

.. |cls_wd_cse| image:: ../experiments/figures/hyppar_tuning/finetune_wd_cse.png
    :width: 300

.. |cls_dr_bpe| image:: ../experiments/figures/hyppar_tuning/finetune_dr_bpe.png
    :width: 300

.. |cls_dr_cse| image:: ../experiments/figures/hyppar_tuning/finetune_dr_cse.png
    :width: 300
    
.. |cls_lr_prb| image:: ../experiments/figures/hyppar_tuning/probing_lr_cse.png
    :width: 300
    
.. |cse_linear| image:: ../experiments/figures/hyppar_tuning/cse_linear.png
    :width: 220

.. |cse_lin-rel_mlm| image:: ../experiments/figures/hyppar_tuning/cse_linear_relu_mlm.png
    :width: 220

.. |cse_lin-rel_cls| image:: ../experiments/figures/hyppar_tuning/cse_linear_relu_cls.png
    :width: 220

.. |cse_kernels| image:: ../experiments/figures/hyppar_tuning/cse_kernels.png
    :width: 220

.. |cse_mask_sizes| image:: ../experiments/figures/hyppar_tuning/cse_mask_sizes.png
    :width: 220

.. |cse_tryouts| image:: ../experiments/figures/hyppar_tuning/cse_tryouts.png
    :width: 220

'''