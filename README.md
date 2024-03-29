# Novelty Characterization Using Hierarchical Clustering


An unsupervised learning method for novelty detection and characterization based on qualitative spatial relations

## Requirements

Python 3.6+

## Run the pipeline in sequence:

- **GetQSRRelations.py**: Generate 4 QSR features to present the state transition, including RCC, QDC, STAR-4, and QTC.
- **Existence.py**: Generate one additional existence feature.
- **Read_Data_consequence.py**: Filter state transitions that at least one feature has changed.
- **Data_preprocessing.py**: Concatenate state traisitions and convert them into clustering-welcomed format.
- **Hierarchical_clustering.py**: Perform hierarchical clustering on the state transitions.

## Citing this Work

If you find this method useful, please cite:

@inproceedings{KR2021-43,<br>
    title     = {{Unsupervised Novelty Characterization in Physical Environments Using Qualitative Spatial Relations}},<br>
    author    = {Li, Ruiqi and Hua, Hua and Haslum, Patrik and Renz, Jochen},<br>
    booktitle = {{Proceedings of the 18th International Conference on Principles of Knowledge Representation and Reasoning}},<br>
    pages     = {454--464},<br>
    year      = {2021},<br>
    month     = {11},<br>
    doi       = {10.24963/kr.2021/43},<br>
    url       = {https://doi.org/10.24963/kr.2021/43},<br>
  }


