FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data/processed/processed.csv data/processed/processed.csv
COPY results/raw/test_extra.txt results/raw/test_extra.txt
COPY results/processed/test_eval.csv results/processed/test_eval.csv
COPY saved_models/ saved_models/
COPY src/ src/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]