model="CMLP"
for (( depth=1; depth!=13; depth++ ))
do
  echo "$model depth $depth"
  device=$((($depth - 1) % 4))
  ipython train_NMLP.py $model $depth $device &
done
