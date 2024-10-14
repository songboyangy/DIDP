import time
def set_config(args):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    args.prefix = f'{args.prefix}_{args.data_name}_DDiff_{timestamp}'
    args.model_path = f"save_models/{args.prefix}"

    args.log_path = f"log/{args.prefix}.log"
    return args
