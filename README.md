# Report Topic Classification

This repository is used to train and validate a labeling model for sustainability reports. For more information visit https://github.com/bizres or https://www.businessresponsibility.ch/

The repository contains a streamlit app which can be used to combine different parameters and assess the models performance on the validation data.

The validation of a model in the streamlit app stores the model as pickle file which is then used in the live application.

## About the model, training and testing data

### Data
The ground truth to train and test the model consists of ca. 750 texts which were annotated with the labels *human_rights*, *environment*, *corruption*, *social_concerns* and *employee_concerns*. Some of the texts are not unique if they have multiple labels.

The labels were created by the dev team of bizres, based on relevant extracts from sustainability reports

### Models
The following approaches are tested and validated:

> **Cleansing and tokenization**: The spacy md language model is used for filtering stopwords and lemmatization

> **Text Vectorization**: Bag-of-Words and vectorization using the spacy language model

> **Prediction Algorithm**: Logistic Regression and Naive Bayes

> **Optimization**: Optimizing the modell for overall accuracy or for recall


## Running the application

### Kicak with Docker
- To run the application with Docker, install the [Docker Engine](https://docs.docker.com/engine/install/) on your development machine.

- Navigate to the local repository

- Run ```docker build -t report-topic-classification .``` to build a local docker image

- Run ```docker run -p 8501:8501 report-topic-classification``` to run the docker image

- Visit localhost:8501 to interact with the streamlit app

### Local for development purposes

- For development we use Ubuntu 20.04 with Python 3.9. But the code should also run on Windows with Python 3.9
- If necessary, install Python 3.9 and Pipenv and spawn a virtual environment with ```pipenv shell```
- Install libraries using the Pipfile with ```pipenv install```
- To run the streamlit app usw ```streamlit run app_model_validation.py```
- Have fun developing ;-)