#using python 3.11 and the pinned requirements it should work
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-py311-pinned.txt setup.sh ./
# Pinned versions reproduce the original deployment; alpaca is installed
# without its stale dependency pins (see setup.sh for why).
RUN bash setup.sh --pinned
COPY . .
# copy all project files to all
ENV PYTHONUNBUFFERED=1
#env pythonunbuffer=1 sends python print output directly to output to avoid dockerlogs being delayed or empty
CMD ["python","run.py"]
