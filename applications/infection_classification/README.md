# Infection Classification Model

This repository contains the code for developing the infection classification model used in the infection phenotyping project. Infection classification models can be trained on human annotated ground truth with fluorescence sensor channel and phase channel to predict the state of infection of single cells. The pixels are predicted to be background (class 0), uninfected (class 1) or infected (class 2) by the model.

## Overview

The following scripts are available:

Training: `infection_classification_*model.py` file implements a machine learning model for classifying infections based on various features. The model is trained on a labeled dataset, with fluorescence and label-free images.

Testing: `infection_classifier_testing.py` file tests the 2D infection classification model trained on a 2D dataset.

Prediction: `predict_classifier_testing.py` is an example script to perform prediction using 2D data and 2D model. It can be used to predict the infection type for new samples.
