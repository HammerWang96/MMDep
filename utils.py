import pickle
from random import random
import numpy as np
import torch


def load_data(path):
    # dataset:"taobao", t_behavior:"click"
    print('Loading data from {} ...'.format(path))
    #
    with open(path, "rb") as rf:
        seqs = pickle.load(rf)

    max_seq_length = 0  # max_len(window size)
    max_basket_length = 0  # max_nums of item within a day
    train_users = []  # user id
    train_baskets = []  # multi-type behavior sequences of temporal set
    train_days = []  # timestamp
    train_behaviors = []  # behavior types
    train_t_basket = []  # target set
    train_t_day = []  # target time
    train_old_items = []  # interacted items

    test_users = []
    test_baskets = []
    test_days = []
    test_behaviors = []
    test_t_basket = []
    test_t_day = []
    test_old_items = []

    # user_seq = [user_id， multi-type behavior sequences， target set， is_test]
    for user_seq in seqs:
        seq_baskets = []
        seq_days = []
        seq_behaviors = []
        oldItems = []
        user_id = user_seq[0]  # user_id
        baskets = user_seq[1]  # input
        t_basket = user_seq[2]  # output
        is_test = user_seq[3]  # is_test
        if not is_test:
            train_users.append(user_id)
            train_t_basket.append(t_basket['items'])
            train_t_day.append(t_basket['time'])
            max_basket_length = len(t_basket['items']) if len(
                t_basket['items']) > max_basket_length else max_basket_length
            for bj in baskets:  # bj:{'time':'2020-10-10', 'items':[1,2]}
                seq_baskets.append(bj['items'])
                max_basket_length = len(bj['items']) if len(bj['items']) > max_basket_length else max_basket_length
                seq_days.append(bj['time'])
                seq_behaviors.append(bj['behavior'])
                oldItems[0:0] = bj['items']
            train_baskets.append(seq_baskets)
            train_days.append(seq_days)
            train_behaviors.append(seq_behaviors)
            train_old_items.append(oldItems)
            max_seq_length = len(seq_days) if len(seq_days) > max_seq_length else max_seq_length
        else:
            test_users.append(user_id)
            test_t_basket.append(t_basket['items'])
            test_t_day.append(t_basket['time'])
            max_basket_length = len(t_basket['items']) if len(
                t_basket['items']) > max_basket_length else max_basket_length
            for bi in baskets:
                seq_baskets.append(bi['items'])
                seq_days.append(bi['time'])
                seq_behaviors.append(bi['behavior'])
                max_basket_length = len(bi['items']) if len(bi['items']) > max_basket_length else max_basket_length
                oldItems[0:0] = bi['items']
            test_baskets.append(seq_baskets)
            test_days.append(seq_days)
            test_behaviors.append(seq_behaviors)
            test_old_items.append(oldItems)
            max_seq_length = len(seq_days) if len(seq_days) > max_seq_length else max_seq_length
    train_list = [train_users, train_baskets, train_t_basket, train_behaviors, train_days, train_t_day, train_old_items]
    test_list = [test_users, test_baskets, test_t_basket, test_behaviors, test_days, test_t_day, test_old_items]
    return train_list, test_list, max_basket_length, max_seq_length


def create_seq(baskets, basket_days, max_basket_len, max_seq_len):
    basket_list = np.asarray([[[0] * max_basket_len] * max_seq_len] * len(baskets))

    for i in range(len(baskets)):
        ss = baskets[i]
        days = basket_days[i]
        for j in range(len(ss)):
            bat = ss[j]
            day = days[j]
            basket_list[i][max_seq_len - day][:len(bat)] = bat
    return basket_list


# target set
def create_tBasket(tBasket, max_basket_len):
    basket_list = np.asarray([[0] * max_basket_len] * len(tBasket))
    for i in range(len(tBasket)):
        ss = tBasket[i]
        basket_list[i][:len(ss)] = ss
    return basket_list


def seq_batch_generator(click_data, favor_data, cart_data, buy_data, tBuy_data,
                        user_data, old_items, batch_size, shuffle):  # tClkData, tFavData, tCartData,
    total_len = len(user_data)
    if shuffle:
        index = list(range(0, total_len))
        random.shuffle(index)
        click_data = click_data[index]
        favor_data = favor_data[index]
        cart_data = cart_data[index]
        buy_data = buy_data[index]
        tBuy_data = tBuy_data[index]
        user_data = user_data[index]

    batchNum = int(total_len / batch_size)
    if total_len % batch_size != 0:
        batchNum += 1
    for i in range(batchNum):
        first = i * batch_size
        last = (first + batch_size) if (first + batch_size) < total_len else total_len

        yield i, {'clickData': click_data[first:last], 'favorData': favor_data[first:last],
                  'cartData': cart_data[first:last],
                  'buyData': buy_data[first:last], 'tBuyData': tBuy_data[first:last],
                  'userData': user_data[first:last],
                  'oldItemData': np.array([old_items[0][first:last], old_items[1][first:last],
                                           old_items[2][first:last],
                                           old_items[3][first:last]]
                                          )}
def y_onehot(bask_index, NB_ITEMS, device):
    onehot = torch.zeros(bask_index.shape[0], NB_ITEMS + 1, device=device)
    onehot = onehot.scatter_(1, bask_index, 1)
    onehot = onehot[:, 1:]
    return onehot


def findPreAllOldItems(clickOldItems, favorOldItems, cartOldItems, buyOldItems):
    preAllOldItems_clk = []
    preAllOldItems_fav = []
    preAllOldItems_cart = []
    preAllOldItems_buy = []
    maxOldItemsLen_clk = 0
    maxOldItemsLen_fav = 0
    maxOldItemsLen_cart = 0
    maxOldItemsLen_buy = 0
    for i in range(len(clickOldItems)):
        allOldItems_clk = []
        allOldItems_fav = []
        allOldItems_cart = []
        allOldItems_buy = []
        allOldItems_clk[0:0] = clickOldItems[i]
        allOldItems_fav[0:0] = favorOldItems[i]
        allOldItems_cart[0:0] = cartOldItems[i]
        allOldItems_buy[0:0] = buyOldItems[i]
        xxx1 = list(set(allOldItems_clk))
        xxx2 = list(set(allOldItems_fav))
        xxx3 = list(set(allOldItems_cart))
        xxx4 = list(set(allOldItems_buy))
        maxOldItemsLen_clk = len(xxx1) if len(xxx1) > maxOldItemsLen_clk else maxOldItemsLen_clk
        preAllOldItems_clk.append(xxx1)
        maxOldItemsLen_fav = len(xxx2) if len(xxx2) > maxOldItemsLen_fav else maxOldItemsLen_fav
        preAllOldItems_fav.append(xxx2)
        maxOldItemsLen_cart = len(xxx3) if len(xxx3) > maxOldItemsLen_cart else maxOldItemsLen_cart
        preAllOldItems_cart.append(xxx3)
        maxOldItemsLen_buy = len(xxx4) if len(xxx4) > maxOldItemsLen_buy else maxOldItemsLen_buy
        preAllOldItems_buy.append(xxx4)
    maxOldItemsLen = max(maxOldItemsLen_clk, maxOldItemsLen_fav, maxOldItemsLen_cart, maxOldItemsLen_buy)
    return [preAllOldItems_clk, preAllOldItems_fav, preAllOldItems_cart, preAllOldItems_buy], maxOldItemsLen
