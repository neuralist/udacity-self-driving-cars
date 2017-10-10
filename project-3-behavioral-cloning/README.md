![alt text](https://img.shields.io/badge/Course-Udacity--SDC-blue.svg)

# **Project 3: Behavioral Cloning** 

The goal of this project is to teach a simulated car to drive autonomously around a set of race tracks.

![Car simulator](./report_images/lake-track.png "Car simulator")

It is a part of the Udacity nanodegree Self Driving Cars, term 1. 

---

### The Behavioural Cloning Mechanism

##### Description
Details of the behavioral cloning mechanism is described in `Project Report.md`.

### Implementation

##### Scripts
`model.py`: Script to create and train the model.
`drive.py`: Script to drive the car autonomously.

##### Results
`model.h5`: KERAS model, trained on the datasets.
`videos`: Folder with videos showing the driving performance of the car.

### Usage

##### Create and train model
Training parameters can be set:
- _epochs_
- _batch_size_
- _learning_rate_

```sh
python model.py --epochs 25 --batch_size 128 --learning_rate 0.001
```

##### Drive car autonomously
```sh
python drive.py model.h5 
```
After launching the script, start the car simulator application (not included in the repo). Choose a track and press 'Autonomous'. 