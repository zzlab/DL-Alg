for ((epoch=10; epoch != 50; epoch += 2))
# for ((epoch = 1; epoch != 3; epoch += 2))
do
  echo $epoch
  ipython NIN_lr_decay_epoch_0_grid_search.py $epoch
done
