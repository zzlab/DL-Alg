interval=4
for ((i=0; i!=4; i++))
do
  ipython dr-loss-curve.py ReLU $interval $i 1024 1024 1024 1024 10 &
  let interval*=2
done
wait
for ((i=0; i!=4; i++))
do
  ipython dr-loss-curve.py ReLU $interval $i 1024 1024 1024 1024 10 &
  let interval*=2
done
wait
