model="DReLU"
for (( depth=1; depth!=13; depth++ ))
do
  echo "$model depth $depth"
  device=$((($depth - 1) % 4))
  echo $device
  ipython activation_distribution.py $model $depth $device &
done
echo "finished"
