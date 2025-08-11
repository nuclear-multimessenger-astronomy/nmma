FROM condaforge/mambaforge:4.9.2-5 as conda

RUN apt update && apt install gcc -y && apt clean

COPY api/environment.yml .
COPY requirements.txt .

RUN mamba env create -f environment.yml
RUN conda clean -afy
RUN find -name '__pycache__' -type d -exec rm -rf '{}' '+'
RUN rm -rf /opt/conda/envs/nmma/lib/python3.12/site-packages/pip
RUN rm -rf /opt/conda/envs/nmma/lib/python3.12/i{dlelib, ensurepip}
RUN rm -rf /opt/conda/envs/nmma/lib{a,t,l,u}san.so
RUN find -name '*.a' -delete

FROM gcr.io/distroless/base-debian10

COPY --from=conda /opt/conda/envs/nmma /env
COPY /nmma /app/nmma
COPY api/app.py /app/app.py
COPY api/log.py /app/log.py
COPY /priors /app/priors
COPY --from=conda /usr/bin/cat /usr/bin/cat

EXPOSE 6901

CMD ["/env/bin/python", "/app/app.py"]
