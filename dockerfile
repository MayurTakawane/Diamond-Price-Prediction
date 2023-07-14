FROM python:3.8-slim
# copping all files in app folder
COPY . /app
# setting app folder as working directory
WORKDIR /app
# creating package
RUN python setup.py sdist bdist_wheel
# installing requirements
RUN pip install -r requirements.txt
# running app
CMD python app.py