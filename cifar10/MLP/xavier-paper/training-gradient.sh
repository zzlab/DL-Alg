for activation in Tanh ReLU DReLU
do
  for ini_mode in default layer-by-layer
  do
    echo $activation $ini_mode
    ipython training-gradient.py $activation $ini_mode
  done
done
