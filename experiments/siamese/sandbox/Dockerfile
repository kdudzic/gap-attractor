# Grader for LiveCodeBench programs: python 3.12 + numpy (LCB's testing_util imports it), LCB's testing_util (vendored,
# unchanged) and our driver. Runs with no network, a read-only root, as the host user (ctxfile/sandbox.py settings).
# Build from experiments/siamese:  docker build -t siamese-grader:1 -f sandbox/Dockerfile .
# The base is the same pinned image as experiments/context_file/sandbox.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN pip install --no-cache-dir numpy==2.3.4
RUN mkdir -p /grader /work && chmod 1777 /work
COPY lcb/testing_util.py lcb/driver.py /grader/
WORKDIR /work
CMD ["sleep", "infinity"]
