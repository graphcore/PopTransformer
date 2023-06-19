mkdir benchmark

for il in 32 128
do
  for ml in 256 1024
  do
    python inference.py model.input_length=$il model.max_length=$ml > benchmark/log_${il}_${ml}.log
  done
done

