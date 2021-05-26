# Novelty-Characterization-Using-Hierarchical-Clustering


An unsupervised learning method for novelty detection and characterization based on qualitative spatial relations.

## Requirements

Python 3.6+

## Run the pipeline in sequence:
- **Input**: GeoJson formatted groundtruth files.

- **GetQSRRelations.py**: Generate 4 QSR features to present the state transition, including RCC, QDC, STAR-4, and QTC.
- **Existence.py**: Generate one additional existence feature.
- **Read_Data_consequence.py**: Filter state transitions that at least one feature has changed.
- **Data_preprocessing.py**: Concatenate state traisitions and convert them into clustering-welcomed format.
- **Hierarchical_clustering.py**: Perform hierarchical clustering on the state transitions.

- **Output**: TXT format clusters.

## Citing this Work

If you use this method in your research, please cite:

Paper is now under review of KR2021 conference.

