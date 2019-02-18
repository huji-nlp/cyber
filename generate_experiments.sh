#!/usr/bin/env bash

for EXPERIMENT in experiments/ebay_vs_onion_drugs_legal/*; do
  BASENAME=$(basename ${EXPERIMENT})
  for PART in drugs forums; do
      sed "s/onion_${PART}_legal/onion_${PART}_illegal/;s/ebay/onion_${PART}_legal/" ${EXPERIMENT} >\
        experiments/onion_${PART}_legal_vs_illegal/${BASENAME}
  done
done
