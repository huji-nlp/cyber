#!/usr/bin/env bash

SOURCE_DIR=experiments/ebay_vs_onion_drugs_legal
for MODEL in elmoattention gloveboe_averaged gloveboe_sum gloveseq2vec naivebayes; do
  EXPERIMENT=${SOURCE_DIR}/${MODEL}.json
  for MASK in dropcontent dropfunc poscontent posfunc pos; do
    sed "s/\(data\/\S*\)\(\.txt\)/\1.${MASK}\2/g" ${EXPERIMENT} > ${SOURCE_DIR}/${MODEL}_${MASK}.json
  done
done

rm -f experiments.txt
for EXPERIMENT in ${SOURCE_DIR}/*; do
  echo ${EXPERIMENT} >> experiments.txt
  BASENAME=$(basename ${EXPERIMENT})
  for PART in drugs forums; do
    OUT_DIR=experiments/onion_${PART}_legal_vs_illegal
    mkdir -p ${OUT_DIR}
    sed "s/onion_drugs_legal/onion_${PART}_illegal/;s/ebay/onion_${PART}_legal/" ${EXPERIMENT} > ${OUT_DIR}/${BASENAME}
    echo ${OUT_DIR}/${BASENAME} >> experiments.txt
  done
  OUT_DIR=experiments/onion_drugs_legal_vs_illegal_test_forums
  mkdir -p ${OUT_DIR}
  sed "/test/s/drugs/forums/" experiments/onion_drugs_legal_vs_illegal/${BASENAME} > ${OUT_DIR}/${BASENAME}
  echo ${OUT_DIR}/${BASENAME} >> experiments.txt
done
NUM_EXPERIMENTS=$(($(cat experiments.txt | wc -l) - 1))
sed -i "s/--array=0-.*/--array=0-${NUM_EXPERIMENTS}/" train_all.sh