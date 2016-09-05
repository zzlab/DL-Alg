activation="BNDReLU"
for (( depth=3; depth!=11; depth++ ))
do
  echo "$depth layers $activation network"
  ipython MLP_saturation.py -- --activation=$activation --hidden-layers=$depth &
done
wait
echo "finished"
