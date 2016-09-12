model="CMLP"
for (( depth=3; depth!=4; depth++ ))
do
  echo "$model depth $depth"
  ipython train_MLP.py $model $depth
done
echo "finished"
