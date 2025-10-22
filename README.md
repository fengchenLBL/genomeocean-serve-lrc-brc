# GenomeOcean Serve Container

## Building and Running the Container Lawrencium (https://lrc-ondemand.lbl.gov/) / Savio (https://ood.brc.berkeley.edu/)

### Build Apptainer .sif File
#### Set up Apptainer cache directory:
```
mkdir -p /global/scratch/users/$USER/.apptainer
export APPTAINER_CACHEDIR=/global/scratch/users/$USER/.apptainer
```

#### Create a temporary directory for model cache:
```
mkdir -p ./tmp
```

#### Convert the Docker image to .sif:
```
apptainer build genomeocean-serve.sif docker://fengchenlbl/genomeocean-serve:latest
```
* Time: ~20-30 minutes.

#### Verify:
```
ls -lh genomeocean-serve.sif
```
* Expected size: ~16.7GB.

### Start GenomeOcean Server
#### Run with GPU and cache binding:
```
apptainer run --nv --bind ./tmp:/workspace/model_cache genomeocean-serve.sif
```
  * Output should show CUDA initialization and server startup:
```
INFO:     Started server process [80474]
INFO:     Waiting for application startup.
[model download logs...]
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Test Endpoints
#### Check health status (in another terminal):
```
curl -X GET http://localhost:8080/health
```
  * Expected: `{"status":"ready"}`.

#### Sequence Generation:
```
curl -X POST http://localhost:8080/generate -H "Content-Type: application/json" -d '{
"instances": [{
"prompt": "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
"num": 5,
"length": 150,
"temperature": 1.0,
"top_k": 50,
"top_p": 0.9,
"min_new_tokens": 20,
"max_new_tokens": 50,
"do_sample": true,
"early_stopping": false,
"repetition_penalty": 1.0
}]
}'
```
  * Expected: `{"predictions":[{"sequences":["ACCAGGGCCTCACTGGTCACGTAGGGTGTCTGGGGGACGTAGGTGACGATGGGCAGCAGGGTGATGAAGTCCCGGCCCACCAGCTTTTTGTAGACCTGGAACCAGCCGCCGCAGATGATATAGCAGCCGATGCCGGGGATGCGCAGCAGCCGTTTGATGAGCATGTCGTTGACCAGATTGCCCCAGGCATAGGCCACCCGGATGTCCGGGTTTTTGGGGGTGCCGTGGG","..."]}]}`.

#### Embedding:
```
curl -X POST http://localhost:8080/embed -H "Content-Type: application/json" -d '{
"instances": [{"seqs":
["GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
"CAGTCAGTGGCTAGCATGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGGTGCATGCTAGCATCGATCGATCGAA"]}]
}'
```
  * Expected: `{"predictions":[{"embeddings":[[-6.518601417541504,-6.508556365966797,0.2003813236951828,...]]}`.
