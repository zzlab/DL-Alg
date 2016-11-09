for activation in ReLU
do
  # for ((epoch=10; epoch != 50; epoch += 2))
  # for ((epoch = 1; epoch != 3; epoch += 2))
  for ((epoch=50; epoch != 100; epoch += 2))
  do
    echo $epoch
    ipython resnet_lr_decay_epoch_0_grid_search.py $epoch $activation
  done
done
