# for activation in ReLU DReLU
for activation in DReLU
do
  for ini_mode in normal layer-by-layer
  do
    echo NIN $activation $ini_mode
    ipython NIN_ini.py $activation $ini_mode
  done
done
