# search learning rate space
# for lr in 0.3 0.2 0.1 0.05
# do
#   echo "learning rate $lr"
#   ipython MXNIN.py $lr
# done

# search momentum space
for momentum in 0.2 0.5 0.8
do
  echo "momentum $momentum"
  ipython MXNIN.py $momentum
done
