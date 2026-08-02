python enumerate_protostructures.py \
  --random-elements \
  --n-samples 100000 \
  --max-complexity 5 \
  --min-elements 2 \
  --max-elements 5 \
  --seed 123 \
  --output protostructures-100k.txt


python crystal_structure_generator_lhs.py \
  --input protostructures-100k.txt \
  --outdir crystal-structures-100k \
  --workers "$(nproc)" \
  --seed 123

For substantially faster generation, use a fixed 20 LHS samples:
python crystal_structure_generator_lhs.py \
  --input protostructures-100k.txt \
  --outdir crystal-structures-100k \
  --workers "$(nproc)" \
  --n-samples 20 \
  --seed 123
