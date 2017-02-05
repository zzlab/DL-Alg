for N in 3 5 7 9 18
do
  echo $N relu 1
  ipython train_residual_network.py $N relu 1
done

for N in 3 5 7 9 18
do
  echo $N drelu 1
  ipython train_residual_network.py $N drelu 1
  echo $N drelu 2
  ipython train_residual_network.py $N drelu 2
done
