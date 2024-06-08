FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app/src
EXPOSE 5000
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["flask", "run"]