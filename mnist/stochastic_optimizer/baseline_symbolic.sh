GPU=0
for HIDDEN_LAYERS in 4
do
  for ACTIVATION in ReLU
  do
    ipython baseline_symbolic.py $HIDDEN_LAYERS $ACTIVATION $GPU
  done
done
