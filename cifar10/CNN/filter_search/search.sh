for activation in DReLU
do
  for filter in 192
#  for (( filter=16; filter!=336; filter+=16 ))
  do
    echo "$activation $filter"
    ipython search.py $activation $filter 
  done
done
