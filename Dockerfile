FROM mambaorg/micromamba:1.5.1
USER root

COPY env.yaml /tmp/env.yaml
RUN micromamba create -y -f /tmp/env.yaml && \
	micromamba clean --all --yes

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    bc \
                    ca-certificates \
                    curl \
                    git \
                    gnupg \
                    lsb-release \
                    netbase \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ENV FLYWHEEL=/flywheel/v0
WORKDIR ${FLYWHEEL}
ENV PATH="/opt/conda/envs/wbhi-redcap/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/wbhi-redcap/bin/pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p ${FLYWHEEL}
COPY run.py ${FLYWHEEL}/run.py
WORKDIR ${FLYWHEEL}

ENTRYPOINT ["/flywheel/v0/run.py"]
