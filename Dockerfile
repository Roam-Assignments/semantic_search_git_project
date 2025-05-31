FROM python:3.10

WORKDIR /app

# Install app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Clone the semantic-db repo
RUN git clone https://github.com/Roam-Assignments/semantic-db.git

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
