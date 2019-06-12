# CNJC ImageOpt
## Eshed Margalit
#### eshedm@stanford.edu

Notebooks to demonstrate image optimization methods.

### Table of Contents
1. 01_Sanity_Check.ipynb
    
    For a given unit in `conv1` (for which the true RF is known) demonstrates direct optimization for maximization of that unit's activity.

1. 02_Single_Unit.ipynb
    
    For a unit in `conv4`, perform the same sort of optimization as in the previous notebook.

1. 03_Single_Unit_Preproc.ipynb
    
    Demonstrates the advantages of preprocessing steps such as jitter and random rotation.

1. 04_Channel_Preproc.ipynb
    
    Instead of optimizing for a single unit, optimizes for high activity in an entire convolutional channel (all spatial taps at once)

1. 05_Channel_Preproc_TotalVariation.ipynb
    
    Improves on preprocessing by also penalizing large pixel-to-pixel differences in the image at each step

1. 06_Channel_Preproc_Simulated_Mapping.ipynb
    
    Finally, demos an example of what it would be like to optimize for a downstream neuron, where the mapping from a layer's activity to that neuron is known.