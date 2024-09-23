import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-preprocess', action='store_true', help="preprocess dataset")   #when you first run code, you should set it to true.

##### model parameters
parser.add_argument('-data_name', type=str, default='twitter', choices=['weibo22', 'memes', 'twitter', 'douban'], help="dataset")
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-max_lenth', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-posSize', type=int, default=8, help= "the position embedding size")
parser.add_argument('--embSize', type=int, default=64, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=float, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default= 0.001, help='ssl graph task maginitude')  
parser.add_argument('--beta2', type=float, default=0.005, help='ssl cascade task maginitude')  
parser.add_argument('--window', type=int, default=10, help='window size')  
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.2)
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--prefix', type=str, default='test', help='prefix to name a trial')
parser.add_argument('--tau', type=float, default=0.5, help='temperature of ssl')
parser.add_argument('--ssl_alpha', type=float, default=0.01, help='coefficient of ssl')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--save_model', action='store_true', help="save model")
parser.add_argument('--inter', action='store_true', help="inter or interAintra")

#####data process
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)

###save model
parser.add_argument('-save_path', default= "./checkpoint/")
parser.add_argument('-patience', type=int, default=10, help="control the step of early-stopping")
# diff reverse params (DNN)
parser.add_argument('--dims', type=str, default='[200,600]', help='the dims for the DNN')
parser.add_argument('--act', type=str, default='tanh', help='the activate function for the DNN')
parser.add_argument('--w_dims', type=str, default='[200,600]', help='the dims for the W DNN')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--diff_lr', type=float,default=0.001, help="the learning rate")
# diff params
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=5e-3, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.005, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
parser.add_argument('--alpha', type=float, default=0.1, help='balance rec loss and reconstruct loss')
