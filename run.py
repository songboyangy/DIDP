import datetime
import Constants
from tqdm import tqdm
from models.models import *
from torch.utils.data import DataLoader
from dataLoader import datasets, Read_data, Split_data
from utils.parsers import parser
from utils.Metrics import Metrics
from utils.EarlyStopping import *
from utils.graphConstruct import ConHypergraph, ConRelationGraph
from models.DNN import SDNet
import models.diffusion_process as gd
from torch import nn, optim
import logging
from utils.utils import set_config

metric = Metrics()
opt = parser.parse_args()
opt=set_config(opt)

def init_seeds(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def model_training(model, train_loader, val_loader, test_loader, social_graph, opt, social_reverse_model, cas_reverse_model, diffusion_model, logger):
    ''' Model training '''
    model.train()
    social_reverse_model.train()
    cas_reverse_model.train()
    loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    early_stopping = EarlyStopping(patience=opt.patience,args=opt, verbose=True, path=opt.model_path)
    logger.info(opt)

    best_results = {}
    top_K = [10, 50, 100]
    validation_history = 0
    opt_model = optim.Adam(model.parameters(), lr=opt.lr)
    opt_cas_dnn = optim.Adam(cas_reverse_model.parameters(), lr=opt.diff_lr)

    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        total_loss = 0.0
        n_total_words = 0.0
        n_total_correct = 0.0
        recons_loss_list=[]
        logger.info(f'Epoch {epoch} start training:')

        model.train()
        for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(tqdm(train_loader, total=len(train_loader),ncols=80)):
            n_words = label.data.ne(Constants.PAD).sum().float().item()
            n_total_words += n_words

            cascade_item = trans_to_cuda(cascade_item.long(), device_id=opt.device)
            label=label.to(opt.device)
            tar = trans_to_cuda(label.long(), device_id=opt.device)
            cascade_time = trans_to_cuda(cascade_time.long(), device_id=opt.device)
            label_time = trans_to_cuda(label_time.long(), device_id=opt.device)
            # pred = model(cascade_item, social_graph, diffusion_model, social_reverse_model,
            #              cas_reverse_model)

            pred, recons_loss,ssl = model(cascade_item,cascade_time,label,  social_graph, diffusion_model, cas_reverse_model)

            recons_loss_list.append(recons_loss.item())
            loss, n_correct = get_performance(loss_function, pred, tar)
            #loss应该是有问题的,loss出现了严重的不平衡问题
            #loss = (1 - opt.alpha) * loss + opt.alpha * recons_loss
            loss=loss/n_words
            loss =  loss +  opt.diff_alpha*recons_loss+ opt.ssl_alpha*ssl
            #loss = loss + recons_loss

            if torch.isinf(loss).any():
                logger.warning('Encountered NaN/Inf loss')

            opt_model.zero_grad()

            opt_cas_dnn.zero_grad()

            loss.backward()

            opt_cas_dnn.step()
            opt_model.step()

            total_loss += loss.item()
            n_total_correct += n_correct


        #logger.info('Epoch %d - Total Loss: %.3f', epoch, total_loss)
        average_loss = total_loss / len(train_loader)
        val_scores, val_accuracy = model_testing(model, val_loader, social_graph, social_reverse_model, cas_reverse_model, diffusion_model,loss_function)
        test_scores, test_accuracy = model_testing(model, test_loader, social_graph, social_reverse_model, cas_reverse_model, diffusion_model,loss_function)
        val_loss=val_scores['loss']
        logger.info(f'Train loss {average_loss} recon loss: {np.mean(np.array(recons_loss_list))} Val loss {val_loss}')
        val_scores.pop('loss', None)

        # if validation_history >= val_scores['loss']:
        #     validation_history = val_scores['loss']
        if validation_history <= sum(val_scores.values()):
            validation_history = sum(val_scores.values())
            for K in top_K:
                test_scores['hits@' + str(K)] = test_scores['hits@' + str(K)] * 100
                test_scores['map@' + str(K)] = test_scores['map@' + str(K)] * 100
                best_results['metric%d' % K][0] = test_scores['hits@' + str(K)]
                best_results['epoch%d' % K][0] = epoch
                best_results['metric%d' % K][1] = test_scores['map@' + str(K)]
                best_results['epoch%d' % K][1] = epoch

            logger.info(f'Epoch {epoch} - Average Train Loss: {average_loss:.3f}')
            val_scores_str = '  '.join(f'{metric}: {val_scores[metric] * 100:.3f}%' for metric in val_scores)
            logger.info(f" - Validation scores:\n  - (Validation) Accuracy: {100 * val_accuracy:.3f} %\n  - {val_scores_str}")

            logger.info(" - Test scores:")
            logger.info(f'  - (Testing) Accuracy: {100 * test_accuracy:.3f} %')
            for K in top_K:
                logger.info(f'  - Train Loss: {total_loss:.4f}, Hit@{K}: {best_results[f"metric{K}"][0]:.4f}, '
                            f'MAP@{K}: {best_results[f"metric{K}"][1]:.4f}, Epoch: {best_results[f"epoch{K}"][0]}, '
                            f'{best_results[f"epoch{K}"][1]}')
        model_list=[model,social_reverse_model,cas_reverse_model,diffusion_model]
        early_stopping(-sum(list(val_scores.values())), model_list,logger)
        if early_stopping.early_stop:
            logger.info("Early Stopping")
            break

    return best_results
def model_testing(model, test_loader, social_graph, social_reverse_model, cas_reverse_model, diffusion_model,loss_function, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0.0
    n_correct = 0.0
    #print('start predicting: ', datetime.datetime.now())
    model.eval()
    social_reverse_model.eval()
    cas_reverse_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(tqdm(test_loader,ncols=80)):
            n_words = label.data.ne(Constants.PAD).sum().float().item()
            cascade_item = trans_to_cuda(cascade_item.long(), device_id=opt.device)
            cascade_time = trans_to_cuda(cascade_time.long(), device_id=opt.device)
            label=label.to(opt.device)
            y_pred = model(cascade_item,cascade_time,label,social_graph,diffusion_model,cas_reverse_model,train=False)
            # y_pred = model(cascade_item, social_graph, diffusion_model, social_reverse_model,
            #                                 cas_reverse_model)
            tar = trans_to_cuda(label.long(), device_id=opt.device)
            loss, n_correct = get_performance(loss_function, y_pred, tar)
            loss=loss/n_words
            total_loss += loss.item()

            y_pred = y_pred.detach().cpu()
            tar = label.view(-1).detach().cpu()

            pred = y_pred.max(1)[1]
            gold = tar.contiguous().view(-1)
            correct = pred.data.eq(gold.data)
            n_correct = correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

            scores_batch, scores_len = metric.compute_metric(y_pred, tar, k_list)
            n_total_words += scores_len

            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        average_loss = total_loss / len(test_loader)
        scores['loss']=average_loss

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores, n_correct / n_total_words


def setup_logging(log_path):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



def main(data_path, seed=2023):
    init_seeds(seed)

    if opt.preprocess:
        Split_data(data_path, train_rate=0.8, valid_rate=0.1, load_dict=False)

    train, valid, test, user_size = Read_data(data_path)
    train_data = datasets(train, opt.max_lenth)
    val_data = datasets(valid, opt.max_lenth)
    test_data = datasets(test, opt.max_lenth)

    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    social_graph = ConRelationGraph(data_path)

    device_string = 'cuda:{}'.format(opt.gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)
    opt.device = device

    opt.n_node = user_size
    HG_Item, HG_User = ConHypergraph(opt.data_name, opt.n_node, opt.window)
    HG_Item = trans_to_cuda(HG_Item, device_id=opt.device)
    HG_User = trans_to_cuda(HG_User, device_id=opt.device)

    model = trans_to_cuda(LSTMGNN(hypergraphs=[HG_Item, HG_User], args=opt, dropout=opt.dropout), device_id=opt.device)
    output_dims = [opt.embSize] + [opt.embSize]
    input_dims = output_dims[::-1]
    social_reverse_model = trans_to_cuda(SDNet(input_dims, output_dims, opt.embSize), device_id=opt.device)
    cas_reverse_model = trans_to_cuda(SDNet(input_dims, output_dims, opt.embSize), device_id=opt.device)

    logger=setup_logging(opt.log_path)

    diffusion_model = gd.DiffusionProcess(opt,  opt.noise_schedule, opt.noise_scale, opt.noise_min,
                                           opt.noise_max, opt.steps, opt.device).to(opt.device)

    best_results = model_training(model, train_loader, val_loader, test_loader, social_graph, opt, social_reverse_model, cas_reverse_model,
                                  diffusion_model,logger)
    top_K = [10, 50, 100]
    logger.info('Best_results：')
    for K in top_K:
        logger.info(f'Hit@{K}: {best_results[f"metric{K}"][0]:.4f}, '
                    f'MAP@{K}: {best_results[f"metric{K}"][1]:.4f}, '
                    f'Epoch: {best_results[f"epoch{K}"][0]}, {best_results[f"epoch{K}"][1]}')


if __name__ == "__main__":
    #原来seed是2023
    main(opt.data_name, seed=opt.seed)

