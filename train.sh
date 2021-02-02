nohup python VIFNet16.py --device_id 0 --batch_size 128 --lr 2e-5 --crop 32 --stride 16 --model_name 32_16   > log/32_16.log 2>&1 &
nohup python VIFNet16.py --device_id 1 --batch_size 64 --lr 2e-5 --crop 64 --stride 32 --model_name 64_32   > log/64_32.log 2>&1 &
nohup python VIFNet16.py --device_id 2 --batch_size 32 --lr 2e-5 --crop 128 --stride 32 --model_name 128_32 > log/128_32.log 2>&1 & 


nohup python VIFNet.py --device_id 0 --batch_size 32 --lr 2e-5 --model_name vif > log/vif.log 2>&1 & 
