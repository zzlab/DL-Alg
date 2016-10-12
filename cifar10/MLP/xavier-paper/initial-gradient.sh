for activation in Tanh ReLU
do
  for ini_mode in default layer-by-layer
  do
    echo $activation $ini_mode
    ipython initial-gradient.py $activation $ini_mode
  done
done
