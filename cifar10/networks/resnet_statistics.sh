for prefix in tuned-ReLU-resnet tuned-resnet-statistics untuned-resnet-statistics
do
  ipython resnet_statistics.py $prefix
done
