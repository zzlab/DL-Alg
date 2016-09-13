model="CMLP"
for (( depth=1; depth!=13; depth++ ))
do
  echo "$model depth $depth"
  device=$((($depth - 1) % 4))
  echo $device
  ipython error_of_mean_affine.py $model $depth $device &
done
echo "finished"
