#!/usr/bin/env bash

trap 'kill $(jobs -p)' EXIT  # Kill server on exit
python -m allennlp.service.server_simple \
    --archive-path models/drugs_bcn_with_cuda_0/model.tar.gz \
    --predictor attention_classifier \
    --include-package cyber \
    --title "Cyber Classification Demo" \
    --field-name text_input \
    --port 8001 &
cd demo
#npm install
npm start
