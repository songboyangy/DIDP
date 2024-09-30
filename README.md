# DDiff
Denoising diffusion model for information diffusion prediction

## 参数
标准
```shell
python run.py -data_name douban --prefix exp --gpu 3 -batch_size 64 --lr 0.01 --diff_lr 0.0001 --steps 100 --sampling_steps 40 --noise_scale 0.1 --ssl_alpha 0.1 --inter

python run.py -data_name android --prefix exp --gpu 4 -batch_size 64 --lr 0.001 --diff_lr 0.0001 --steps 50 --sampling_steps 10 --noise_scale 0.1 --ssl_alpha 0.1 --inter -preprocess
```
调参用


```shell
python run.py -data_name twitter --prefix exp --gpu 2 -batch_size 64 --lr 0.001 --diff_lr 0.0001 --steps 50 --sampling_steps 10 --noise_scale 0.1 --ssl_alpha 0.1 --inter --diff_alpha 1
```