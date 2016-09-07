for activation in BGNDReLU
do
  for (( depth=1; depth!=11; depth++ ))
  do
    echo "CNN $activation depth $depth"
    ipython MXCNN.py -- --activation=$activation --convolution-layers=$depth
  done
done
