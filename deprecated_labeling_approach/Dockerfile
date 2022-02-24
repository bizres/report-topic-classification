FROM python:3.9.6

COPY Pipfile* ./

RUN ["pip", "install", "pipenv"]

RUN ["pipenv", "install", "--system", "--deploy", "--ignore-pipfile"]

COPY . .

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run"]
CMD ["app_model_validation.py"]