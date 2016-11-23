device=0
for a in ReLU BNReLU ReLUBN
# for a in Tanh BNTanh TanhBN
do
  ipython MLP-initial-parameters.py $a 1024 1024 1024 1024 10
  ipython train_MLP.py $a $device 1024 1024 1024 1024 10 &
  let device+=1
done
wait
