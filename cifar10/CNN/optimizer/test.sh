for activation in ReLU BNReLU
do
  for (( depth=1; depth!=11; depth++ ))
  do
    echo "CNN $activation depth $depth"
    ipython CNN_optimizer_test.py $activation $depth Adam
  done
done
