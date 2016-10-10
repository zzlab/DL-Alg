for activation in BNDReLU
do
  for (( filter=16; filter!=336; filter+=16 ))
  do
    echo "$activation $filter"
    ipython search.py $activation $filter 
  done
done
