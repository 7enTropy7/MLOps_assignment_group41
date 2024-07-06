FROM python:3.9-slim
COPY . .
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade pip
RUN pip install flask joblib numpy scikit-learn
CMD ["app.py"]
ENTRYPOINT ["python"]
# docker build . -t mlp-flask-app
# docker run -p 8080:8080 mlp-flask-app