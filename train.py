import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import load_npz
from torch import optim
from logger import Logger
from metrics import Recall_at_k_season_batch, NDCG_binary_at_k_season_batch, HitRate_at_k_season_batch
from model import Ours
from utils import load_data, create_seq, create_tBasket, seq_batch_generator, y_onehot, findPreAllOldItems
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay (L2 loss on parameters).')  #
parser.add_argument('--dataset', type=str, default='feizhu', help='the dataset to run')
parser.add_argument('--batch_size', type=int, default=150, help='the batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--cuda', type=int, default=0, help='the number of cuda')
parser.add_argument('--NB_USERS', type=int, default=8277, help='the number of users')
parser.add_argument('--NB_ITEMS', type=int, default=8731, help='the number of items')
parser.add_argument('--use_loss_flag', type=bool, default=False, help='use new loss or not')
parser.add_argument('--emb_hidden', type=int, default=400, help='Number of item units')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--seed', type=int, default=1013, help='Random seed.')
parser.add_argument('--l_lens', default=20, type=int)
parser.add_argument('--lamda', default=0.00001, type=float)

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

env_cuda = "cuda:%d" % args.cuda
device = torch.device(env_cuda if torch.cuda.is_available() else "cpu")  #
top_list = [5, 10, 20]  # @topK

# Model and optimizer
model = Ours(args.emb_hidden, args.NB_ITEMS, args.NB_USERS, args.l_lens, device).to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

logger_path = os.path.join('data/' + args.dataset + '/' + args.dataset + '_log',
                           'lr{0}_itemDim{1}_batchSize{2}_loss.txt'.format(
                               args.lr, args.emb_hidden, args.batch_size
                           ))
logger_loss = Logger(logger_path)
logger_path_1 = os.path.join('data/' + args.dataset + '/' + args.dataset + '_log',
                             'lr{0}_itemDim{1}_batchSize{2}_metrics.txt'.format(
                                 args.lr, args.emb_hidden, args.batch_size
                             ))
logger_metrics = Logger(logger_path_1)
model_param_path = os.path.join('data/' + args.dataset + '/' + args.dataset + '_param', 'step_weight.pkl')
lowLossParamPath = os.path.join('data/' + args.dataset + '/' + args.dataset + '_param', 'lowLoss_weight.pkl')

# 加载数据
click_path = "data/" + args.dataset + "/click_data.pkl"
favor_path = "data/" + args.dataset + "/favor_data.pkl"
cart_path = "data/" + args.dataset + "/cart_data.pkl"
buy_path = "data/" + args.dataset + "/buy_data.pkl"
click_m = "data/" + args.dataset + "/sppmi_clk_f.npz"
favor_m = "data/" + args.dataset + "/sppmi_fav_f.npz"
cart_m = "data/" + args.dataset + "/sppmi_cart_f.npz"
buy_m = "data/" + args.dataset + "/sppmi_buy_f.npz"

click_train_list, click_test_list, click_basket_len, click_seq_len = load_data(click_path)  # basket_len一天中某种行为最多的数量
favor_train_list, favor_test_list, favor_basket_len, favor_seq_len = load_data(favor_path)  # seq_len 时间窗口
cart_train_list, cart_test_list, cart_basket_len, cart_seq_len = load_data(cart_path)
buy_train_list, buy_test_list, buy_basket_len, buy_seq_len = load_data(buy_path)
clk_mat = load_npz(click_m).toarray()
favor_mat = load_npz(favor_m).toarray()
cart_mat = load_npz(cart_m).toarray()
buy_mat = load_npz(buy_m).toarray()

# training samples
train_users = buy_train_list[0]
click_train_baskets = create_seq(click_train_list[1], click_train_list[4], click_basket_len,
                                 click_seq_len)
favor_train_baskets = create_seq(favor_train_list[1], favor_train_list[4], favor_basket_len,
                                 favor_seq_len)
cart_train_baskets = create_seq(cart_train_list[1], cart_train_list[4], cart_basket_len, cart_seq_len)
buy_train_baskets = create_seq(buy_train_list[1], buy_train_list[4], buy_basket_len, buy_seq_len)

buy_train_tBask = create_tBasket(buy_train_list[2], buy_basket_len)

trainOldItems, trainOldItemsLen = findPreAllOldItems(click_train_list[6], favor_train_list[6], cart_train_list[6],
                                                     buy_train_list[6])
train_old_items_clk = create_tBasket(trainOldItems[0], trainOldItemsLen)
train_old_items_fav = create_tBasket(trainOldItems[1], trainOldItemsLen)
train_old_items_cart = create_tBasket(trainOldItems[2], trainOldItemsLen)
train_old_items_buy = create_tBasket(trainOldItems[3], trainOldItemsLen)
train_old_items = [train_old_items_clk, train_old_items_fav, train_old_items_cart, train_old_items_buy]

# testing samples
test_users = buy_test_list[0]
click_test_baskets = create_seq(click_test_list[1], click_test_list[4], click_basket_len,
                                click_seq_len)
favor_test_baskets = create_seq(favor_test_list[1], favor_test_list[4], favor_basket_len,
                                favor_seq_len)
cart_test_baskets = create_seq(cart_test_list[1], cart_test_list[4], cart_basket_len, cart_seq_len)
buy_test_baskets = create_seq(buy_test_list[1], buy_test_list[4], buy_basket_len, buy_seq_len)

test_buy_tBask = create_tBasket(buy_test_list[2], buy_basket_len)
testOldItems, testOldItemsLen = findPreAllOldItems(click_test_list[6], favor_test_list[6], cart_test_list[6],
                                                   buy_test_list[6])
test_old_items_clk = create_tBasket(testOldItems[0], testOldItemsLen)
test_old_items_fav = create_tBasket(testOldItems[1], testOldItemsLen)
test_old_items_cart = create_tBasket(testOldItems[2], testOldItemsLen)
test_old_items_buy = create_tBasket(testOldItems[3], testOldItemsLen)
test_old_items = [test_old_items_clk, test_old_items_fav, test_old_items_cart, test_old_items_buy]


def train():
    torch.autograd.set_detect_anomaly(True)
    best_loss = 1e9
    lowLossEpoch = 0
    count = 0
    best_r5 = -1
    best_n5 = -1
    for _, epoch in enumerate(range(1, args.epochs + 1)):
        model.train()
        train_loss = 0

        train_generator = seq_batch_generator(click_train_baskets, favor_train_baskets, cart_train_baskets,
                                              buy_train_baskets, buy_train_tBask, train_users, train_old_items,
                                              args.batch_size, False)
        for batch_id, data in train_generator:
            optimizer.zero_grad()
            clickData = Variable(torch.LongTensor(data['clickData'])).to(device)  # click attention
            favorData = Variable(torch.LongTensor(data['favorData'])).to(device)  # favor attention
            cartData = Variable(torch.LongTensor(data['cartData'])).to(device)
            buyData = Variable(torch.LongTensor(data['buyData'])).to(device)

            tBuyData = Variable(torch.LongTensor(data['tBuyData'])).to(device)  # target basket
            userData = Variable(torch.LongTensor(data['userData'])).to(device)  # user
            oldItemsData = Variable(torch.LongTensor(data['oldItemData'])).to(device)

            buy_score, sim, omiga = model(clickData, favorData, cartData, buyData, userData,
                                          oldItemsData)

            loss = loss_fun_1(tBuyData, buy_score, args.NB_ITEMS, sim, omiga, args.lamda, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_loss = train_loss / (batch_id + 1)

        logger_loss.info("Training | " + 'Epoch: {}'.format(epoch) + ' | Average loss: {:.6f}'.format(avg_loss))

        torch.save(model.state_dict(), model_param_path)
        metrics = test(model_param_path)
        if (metrics['RList'][0] > best_r5 and metrics['NList'][0] - best_n5 > -0.005) or metrics['NList'][0] > best_n5:
            for id in range(len(top_list)):
                logger_metrics.info("Testing | bestMetrics | Epoch: {}".format(epoch) + " | recall@{}".format(
                    top_list[id]) + ": {}".format(metrics['RList'][id]))
                logger_metrics.info("Testing | bestMetrics | Epoch: {}".format(epoch) + " | ndcg@{}".format(
                    top_list[id]) + ": {}".format(metrics['NList'][id]))
                logger_metrics.info(
                    "Testing | bestMetrics | Epoch: {}".format(epoch) + " | HR@{}".format(top_list[id]) + ": {}".format(
                        metrics['HList'][id]))
                if metrics['RList'][0] > best_r5:
                    best_r5 = metrics['RList'][0]
                if metrics['NList'][0] > best_n5:
                    best_n5 = metrics['NList'][0]

        if train_loss < best_loss:
            count = 0
            best_loss = train_loss
            torch.save(model.state_dict(), lowLossParamPath)
            lowLossEpoch = epoch
        else:
            count += 1
            if count == 5:
                best_loss = 0
                logger_loss.info('early stop!')

    metrics = test(lowLossParamPath)
    for id in range(len(top_list)):
        logger_metrics.info(
            "Testing | lowloss | Epoch: {}".format(lowLossEpoch) + " | recall@{}".format(top_list[id]) + ": {}".format(
                metrics['RList'][id]))
        logger_metrics.info(
            "Testing | lowloss | Epoch: {}".format(lowLossEpoch) + " | ndcg@{}".format(top_list[id]) + ": {}".format(
                metrics['NList'][id]))
        logger_metrics.info(
            "Testing | lowloss | Epoch: {}".format(lowLossEpoch) + " | HR@{}".format(top_list[id]) + ": {}".format(
                metrics['HList'][id]))


def loss_fun_1(tData, score, NB_ITEMS, sim, omiga, lmda, device):  #
    y = y_onehot(tData, NB_ITEMS=NB_ITEMS, device=device)

    posLogits = torch.mul(F.log_softmax(score, dim=-1), y)
    loss_ = -torch.sum(posLogits)
    # matrices factorization
    flat = tData.flatten()
    nonzero_indices = torch.nonzero(flat, as_tuple=False)
    ground_true = np.array(flat[nonzero_indices].squeeze(-1).cpu())

    mij_clk = torch.tensor(clk_mat[ground_true, 1:]).to(device)
    mij_fav = torch.tensor(favor_mat[ground_true, 1:]).to(device)
    mij_cart = torch.tensor(cart_mat[ground_true, 1:]).to(device)
    mij_buy = torch.tensor(buy_mat[ground_true, 1:]).to(device)

    clk_mask = (mij_clk != 0).int()
    fav_mask = (mij_fav != 0).int()
    cart_mask = (mij_cart != 0).int()
    buy_mask = (mij_buy != 0).int()

    sims_clk = sim[ground_true - 1]
    sims_fav = sim[ground_true - 1]
    sims_cart = sim[ground_true - 1]
    sims_buy = sim[ground_true - 1]

    wi_clk = omiga[ground_true - 1].repeat(1, NB_ITEMS)
    wi_fav = omiga[ground_true - 1].repeat(1, NB_ITEMS)
    wi_cart = omiga[ground_true - 1].repeat(1, NB_ITEMS)
    wi_buy = omiga[ground_true - 1].repeat(1, NB_ITEMS)

    loss1 = torch.sum(((mij_clk - sims_clk - wi_clk) ** 2) * clk_mask)
    loss2 = torch.sum(((mij_fav - sims_fav - wi_fav) ** 2) * fav_mask)
    loss3 = torch.sum(((mij_cart - sims_cart - wi_cart) ** 2) * cart_mask)
    loss4 = torch.sum(((mij_buy - sims_buy - wi_buy) ** 2) * buy_mask)
    loss = loss_ + (loss1 + loss2 + loss3 + loss4) * lmda

    return loss


def test(param_path):
    # param_path
    model.load_state_dict(torch.load(param_path))
    model.eval()

    buyRecallList = [torch.zeros(args.batch_size, device=device) for _ in range(len(top_list))]
    buyNdcgList = [torch.zeros(args.batch_size, device=device) for _ in range(len(top_list))]
    buyHRList = [torch.zeros(args.batch_size, device=device) for _ in range(len(top_list))]

    with torch.no_grad():
        test_generator = seq_batch_generator(click_test_baskets, favor_test_baskets, cart_test_baskets,
                                             buy_test_baskets, test_buy_tBask,
                                             test_users, test_old_items, args.batch_size, False)
        for batch_id, data in test_generator:
            clickData = Variable(torch.LongTensor(data['clickData'])).to(device)  # click attention
            favorData = Variable(torch.LongTensor(data['favorData'])).to(device)  # favor attention
            cartData = Variable(torch.LongTensor(data['cartData'])).to(device)
            buyData = Variable(torch.LongTensor(data['buyData'])).to(device)

            tBuyData = Variable(torch.LongTensor(data['tBuyData'])).to(device)

            userData = Variable(torch.LongTensor(data['userData'])).to(device)  # user
            oldItemsData = Variable(torch.LongTensor(data['oldItemData'])).to(device)

            buy_score, sim, omiga = model(clickData, favorData, cartData, buyData, userData,
                                          oldItemsData)  # , sim, omiga

            y_buy = y_onehot(tBuyData, NB_ITEMS=args.NB_ITEMS, device=device)

            for topk_idx, topk in enumerate(top_list):
                buyRecallList[topk_idx] = torch.cat(
                    [buyRecallList[topk_idx], Recall_at_k_season_batch(buy_score, y_buy, k=topk)],
                    dim=0)
                buyNdcgList[topk_idx] = torch.cat(
                    [buyNdcgList[topk_idx], NDCG_binary_at_k_season_batch(buy_score, y_buy, k=topk)],
                    dim=0)
                buyHRList[topk_idx] = torch.cat(
                    [buyHRList[topk_idx], HitRate_at_k_season_batch(buy_score, y_buy, k=topk)],
                    dim=0)
        for topk_idx, topk in enumerate(top_list):
            buyRecallList[topk_idx] = buyRecallList[topk_idx][args.batch_size:].to('cpu')
            buyRecallList[topk_idx] = np.nanmean(buyRecallList[topk_idx], axis=-1)  # list(6), float32

            buyNdcgList[topk_idx] = buyNdcgList[topk_idx][args.batch_size:].to('cpu')
            buyNdcgList[topk_idx] = np.nanmean(buyNdcgList[topk_idx], axis=-1)

            buyHRList[topk_idx] = buyHRList[topk_idx][args.batch_size:].to('cpu')
            buyHRList[topk_idx] = np.nanmean(buyHRList[topk_idx], axis=-1)

        buy_metrics = {'RList': buyRecallList, 'NList': buyNdcgList
            , 'HList': buyHRList}
        return buy_metrics
train()
