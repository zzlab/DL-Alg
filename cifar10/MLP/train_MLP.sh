activation=DReLU
for (( depth=1; depth!=13; depth++ ))
do
  echo "$model depth $depth"
  device=$((($depth - 1) % 4))
  ipython train_MLP.py $activation $depth $device &
done
