# Manual for using the server
Run jupyter notebook on the Server
```
nohup jupyter notebook --no-browser --port=8889 > jupyter.out 2>&1 &   
```
Connect jupyter notebook remotely 
```
ssh -N -f -L localhost:8888:localhost:8889 wluo@146.50.28.79
```
Connect to Shuo
ssh wluo@146.50.28.79

Connect to Das4
> User     : wluo
> Password : YLij99ea
```
ssh wluo@fs4.das4.science.uva.nl
```

Connect to GCP 
34.85.51.246 

Train i3d
nohup python train_i3d.py > i3d.out 2>&1 & 
watch -n 0.5 nvidia-smi
34198