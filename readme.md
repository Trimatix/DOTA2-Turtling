This is a copy of a private repository, and so does not contain the project commit history.

# Prediction of the Turtle Strategy in DOTA 2
This repository contains the source code written for completion of my undergraduate dissertation, to recognise occurences of the turtle strategy in DOTA 2. The paper is available to read [on my personal website](https://jasperlaw.dev).

The code provided has been adapted for use with distributed GPU computing. Some minor configuration may be required for use with CPU instead.

### Inspiration

The model was built after the architecture presented in [Adam Katona's brilliant paper on death prediction](https://arxiv.org/abs/1906.03939), which is well worth a read.


### Architecture

The strategy is modelled with a deep, feedforward neural network, constructed with tensorflow 2.3.0:

![Rough architecture sketch](rough_architecture_sketch.png?raw=true "Rough architecture sketch")

Each hero is represented by 21 input features. Each hero's features is fed to a single subnetwork. Since the representations of heroes should be calculated in the same way, the weights of these hero subnetworks are shared - and so the hero representation subnetwork is called the "shared" subnet.

Each hero's 21 outputs from the shared subnetwork are concatenated into one long vector, which is fed into a final subnetwork with 10 outputs. Each output is regression between 0 and 1, predicting the probability that each hero is currently attempting to employ the turtle strategy.

### Results

The network was trained with a small and highly imbalanced dataset of 37 professional Dota 2 matches, and evaluated at an average precision of 9.04%, and a maximum f1 score of 13.13%.

### Data

The model reads either one or a set of CSV files, where columns are features, and rows are game state readings for a single game tick.

This data was extracted from DOTA 2 replay files using [the Clarity parser](https://github.com/skadistats/clarity), my extension of which is provided in this repository. What is not provided here is the labelling process, which is manual. CSVs must end with 10 label columns, `isTurtling0`, `isTurtling`, ... `isTurtling9`.

The pre-parsed dataset used during my dissertation project is available [here](https://drive.google.com/drive/folders/1y8d6Tg5yoOP-5FK4eDECRm4n1Woh_T6X?usp=sharing).
