initialization=lower-0-upper-1
for activation in BNReLU BNDReLU
do
  ipython MXResNet.py $activation 6 single resnet-$activation-6-single-Adam-$initialization
done
for activation in BNReLU BNDReLU
do
  for kernel in single double
  do
    ipython MXResNet.py $activation 3 kernel resnet-$activation-3-$kernel-Adam-$initialization
  done
done
