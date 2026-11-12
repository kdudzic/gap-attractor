# Sandbox for the bash edit interface: python3, bash, coreutils, sed, grep. No network at run time, read-only root,
# run as the host user (see ctxfile/sandbox.py). Build: docker build -t gap-ctx-sandbox:1 experiments/context_file/sandbox
# The base is pinned by digest (python:3.12-slim-bookworm as of 14 Oct 2026); runs record the built image's id.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN mkdir -p /ctx /work && chmod 1777 /ctx /work
WORKDIR /work
CMD ["sleep", "infinity"]
