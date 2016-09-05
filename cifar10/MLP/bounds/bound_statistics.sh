activation="DReLU"
for (( depth=3; depth!=11; depth++ ))
do
  ipython bound_statistics.py -- --activation=$activation --hidden=$depth &
done
wait
echo "finished"
