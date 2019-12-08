ECE 598 SG Final project: Lane-change Prediction Based on LSTM
==============================================================

Introduction
------------
This serves as the final project for ECE 598 SG. Lane change prediciton based on LSTM using HighD dataset.

Directory Structure
-------------------
```
lane-change-prediction-lstm/ .... Top src dir
|-- LICENSE ..................... Full license text
|-- main.py ..................... main file executing data processing module
|-- process_data.py ............. data processing library
|-- read_data.py ................ read data functionality for data processing
|-- rnn_model.py ................ LSTM training and testing
|-- requirement.txt ............. Requirements for building
```

Usage
-----
1. Please put the HighD dataset ``data/`` in the directory.

2. Run ``python3 main.py`` to process the dataset and the data will be stored into ``output/`` in pickle format.

3. Run ``python3 rnn_model`` to perform the training and testing.


Requirements
------------
Packages installation guide: ``pip3 install -r requirement.txt``

Please use Python3

Notice
------
All rights reserved.