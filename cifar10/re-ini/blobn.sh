device=0
for method in neither_shared shared_mean shared_deviation shared_mean_shared_deviation
do
  echo $method
  ipython blobn.py $method $device &
  let device+=1
done
wait
