#!/usr/bin/env bash

for EXPERIMENT in experiments/ebay_vs_onion_drugs_legal/*; do
  BASENAME=$(basename ${EXPERIMENT})
  for PART in drugs forums; do
    OUTDIR=experiments/onion_${PART}_legal_vs_illegal
    mkdir -p ${OUTDIR}
    sed "s/onion_drugs_legal/onion_${PART}_illegal/;s/ebay/onion_${PART}_legal/" ${EXPERIMENT} > ${OUTDIR}/${BASENAME}
  done
done
