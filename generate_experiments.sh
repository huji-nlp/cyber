#!/usr/bin/env bash

rm -f experiments.txt
for EXPERIMENT in experiments/ebay_vs_onion_drugs_legal/*; do
  echo ${EXPERIMENT} >> experiments.txt
  BASENAME=$(basename ${EXPERIMENT})
  for PART in drugs forums; do
    OUTDIR=experiments/onion_${PART}_legal_vs_illegal
    mkdir -p ${OUTDIR}
    sed "s/onion_drugs_legal/onion_${PART}_illegal/;s/ebay/onion_${PART}_legal/" ${EXPERIMENT} > ${OUTDIR}/${BASENAME}
    echo ${OUTDIR}/${BASENAME} >> experiments.txt
  done
  OUTDIR=experiments/onion_drugs_legal_vs_illegal_test_forums
  mkdir -p ${OUTDIR}
  sed "/test/s/drugs/forums/" experiments/onion_drugs_legal_vs_illegal/${BASENAME} > ${OUTDIR}/${BASENAME}
  echo ${OUTDIR}/${BASENAME} >> experiments.txt
done
