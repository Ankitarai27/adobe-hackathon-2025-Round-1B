FROM python:3.10-slim                  

ENV TRANSFORMERS_OFFLINE=1             
ENV HF_DATASETS_OFFLINE=1              
WORKDIR /app                           

COPY . .                               
RUN pip install --no-cache-dir -r requirements.txt  

CMD ["python", "run.py"]              
