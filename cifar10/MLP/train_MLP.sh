activation="NDReLU"
for (( depth=3; depth!=4; depth++ ))
do
  echo "MLP depth $depth"
  ipython train_MLP.py $activation $depth &
  sleep 20
done
wait
echo "finished"
