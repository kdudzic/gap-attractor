# Grader for CodeEval-Pro / BigCodeBench-Lite-Pro programs: python 3.10 + the libraries the 57 BigCodeBench-Lite-Pro
# pairs import (problems, reference solutions, CodeEval-Pro test code, BigCodeBench unit tests), at BigCodeBench's
# pinned evaluation versions (bigcode-project/bigcodebench Requirements/requirements-eval.txt, commit 09dd993; `regex`
# is not pinned there). HumanEval-Pro needs only the standard library. Our driver: siam/cep_driver.py.
# Runs with no network, a read-only root, as the host user (ctxfile/sandbox.py settings).
# Build from experiments/siamese:  docker build -t siamese-cep:1 -f third_party/codeeval_pro/Dockerfile .
FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
RUN pip install --no-cache-dir numpy==1.21.2 scipy==1.7.2 pandas==2.0.3 matplotlib==3.7.0 seaborn==0.13.2 \
        scikit-learn==1.3.1 statsmodels==0.14.0 holidays==0.29 Faker==20.1.0 python-dateutil==2.9.0 \
        pytz==2023.3.post1 regex==2024.11.6
RUN mkdir -p /grader /work && chmod 1777 /work
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1
COPY siam/cep_driver.py /grader/
WORKDIR /work
CMD ["sleep", "infinity"]
