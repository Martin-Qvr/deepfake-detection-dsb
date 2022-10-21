# Deepfakes detection - 3 mousketeers

This work is the final work on Deepfake detection by the group 3 Musketeers.
The members of the group are :
- Sarah Mayer
- Charles Proye
- Nathan Aïm
- Charles Gleizes
- Martin Quievre

The assignement was to create a Deepfake image classifying model using Deep Learning. 

The emphasis was on good code practice and collaboration, leveraging the features of gitlab and 
introducing unit tests to our model.

The structure of the repo is the following:
- /code/model with all the models, pipeline and predictions 
- /code/exploratory work with the different models we tested
- /code/test, unit tests to test the right behavior of the code.
- /Docs, documentation for our work

Upon training, our custom deep learning model completed the following performance on our holdout validation set (1500 images):

| F1 Score | Recall | Precision | Accuracy |
|---|---|---|---|
| 0.77 | 0.76 | 0.77 | 0.76 |
