# Report Topic Classification

This repository contains the code to generate keywords for topics based on Wikipeida articles and then find the keywords in text data. This code is used to assess sustainability reports for the [Prototype Fund](https://prototypefund.opendata.ch/en/) project businessresponsibility.ch For more information visit https://github.com/bizres or https://www.businessresponsibility.ch/

# If you are interested in the project...

... go to our [main repository](https://github.com/bizres) to read up on our project.

# If you are interested in the code...

... look at our [Jupyter Notebook](https://github.com/bizres/report-topic-classification/blob/master/usage_example.ipynb) to see how we use Wikipeida articles to generate topic keywords and detect topics in sustainability reports.

## Legacy
Before coming up with the approach to use Wikipeida knowledge to detect sustainability topics, we tried an approach to train a model based on pre-labeled data. This approach did not bear fruit because i) there does not exist a training set suitet for the task ii) we did not have the ressources to build up such a training set, iii) the topics we are looking for are too fuzzy to come up with a clear cut labeling approach and iv) we needed an approach which can dynamically look for topics which may emerge in the future. For documentation reasons, the folder [deprecated_labeling_approach](https://github.com/bizres/report-topic-classification/tree/master/deprecated_labeling_approach) contains the project which follows the pre-labeling-approach. The folder contains its own README whcih explains the structure of the project.
