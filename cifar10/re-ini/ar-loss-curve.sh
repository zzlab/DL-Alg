interval=1
for ((i=0; i!=4; i++))
do
  ipython ar.py $interval $i &
  let interval*=2
done
wait
for ((i=0; i!=4; i++))
do
  ipython ar.py $interval $i &
  let interval*=2
done
wait
