import math
import torch
from torch import nn
from mini_module import BWFeedForward1, BWFeedForward2, Gate1, Gate2  #
from utils import y_onehot
import torch.nn.functional as F
from torch.autograd import Variable


class Ours(nn.Module):
    def __init__(self, emb_hidden, NB_ITEMS, NB_USERS, l_lens, device):
        super(Ours, self).__init__()
        self.device = device
        self.NB_ITEMS = NB_ITEMS
        self.NB_USERS = NB_USERS
        self.l_len = l_lens


        self.item_emb = nn.Embedding(NB_ITEMS + 1, emb_hidden, padding_idx=0,
                                     _weight=nn.init.normal_(torch.empty(NB_ITEMS + 1,
                                                                         emb_hidden), 0, 0.1))


        self.user_emb = nn.Embedding(NB_USERS, emb_hidden,
                                     _weight=nn.init.normal_(torch.empty(NB_USERS, emb_hidden), 0, 0.1))  #

        # ema param
        self.user_w1 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))  #
        self.user_d1 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        self.user_w2 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        self.user_d2 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        self.user_w3 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        self.user_d3 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        self.user_w4 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        self.user_d4 = nn.Embedding(NB_USERS, 1,
                                    _weight=nn.init.normal_(torch.empty(NB_USERS, 1), 0, 0.1))
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)

        self.omiga = nn.Embedding(NB_ITEMS + 1, 1, padding_idx=0)
        with torch.no_grad():
            self.omiga.weight[0].fill_(0)

        # multi-behavior gate
        self.w = nn.Parameter(torch.empty(size=(4, 4)))
        nn.init.xavier_uniform_(self.w, gain=1.414)

        # long- and short-term gate
        self.alpha = nn.Parameter(torch.tensor([0.01, 0.01]), requires_grad=True)
        self.gate1 = Gate1(emb_hidden * 2, 1)

        # attention transformation
        self.w1 = nn.Linear(emb_hidden, emb_hidden)
        self.fai1 = nn.ReLU()
        self.w2 = nn.Linear(emb_hidden, emb_hidden)
        self.fai2 = nn.ReLU()
        self.w3 = nn.Linear(emb_hidden, emb_hidden)
        self.fai3 = nn.ReLU()
        self.w4 = nn.Linear(emb_hidden, emb_hidden)
        self.fai4 = nn.ReLU()

        self.u_ = nn.Linear(emb_hidden, self.NB_ITEMS, bias=True)

    def forward(self, clickData, favorData, cartData, buyData, userData, oldItemsData):
        # embedding层
        e_clickData = self.item_emb(clickData)  # (b, len, c, emb)
        e_favorData = self.item_emb(favorData)
        e_cartData = self.item_emb(cartData)
        e_buyData = self.item_emb(buyData)
        e_user = self.user_emb(userData)  # (b, emb)

        # sum层
        e_clickBasket = self.sumPooling(e_clickData)  # (b, len, emb)
        e_favorBasket = self.sumPooling(e_favorData)
        e_cartBasket = self.sumPooling(e_cartData)
        e_buyBasket = self.sumPooling(e_buyData)

        # long-term-GAT
        mask1 = self.tensorMask(e_clickBasket)  # [:, :self.l_len - self.s_len, :]
        clk_l = self.Attention1(e_user, e_clickBasket, e_clickBasket, mask1)

        mask2 = self.tensorMask(e_favorBasket)
        fav_l = self.Attention2(e_user, e_favorBasket, e_favorBasket, mask2)

        mask3 = self.tensorMask(e_cartBasket)
        cart_l = self.Attention3(e_user, e_cartBasket, e_cartBasket, mask3)

        mask4 = self.tensorMask(e_buyBasket)
        buy_l = self.Attention4(e_user, e_buyBasket, e_buyBasket, mask4)

        # short-term
        # EMA weight
        clk_dweight1 = self.user_w1(userData)
        clk_dweight2 = self.user_d1(userData)
        #
        fav_dweight1 = self.user_w2(userData)
        fav_dweight2 = self.user_d2(userData)
        #
        cart_dweight1 = self.user_w3(userData)
        cart_dweight2 = self.user_d3(userData)
        #
        buy_dweight1 = self.user_w4(userData)
        buy_dweight2 = self.user_d4(userData)

        # multi-scale
        e_clickBasket_s1 = e_clickBasket
        e_favorBasket_s1 = e_favorBasket
        e_cartBasket_s1 = e_cartBasket
        e_buyBasket_s1 = e_buyBasket

        e_clickBasket_s2 = e_clickBasket[:, self.l_len // 2:, :]
        e_favorBasket_s2 = e_favorBasket[:, self.l_len // 2:, :]
        e_cartBasket_s2 = e_cartBasket[:, self.l_len // 2:, :]
        e_buyBasket_s2 = e_buyBasket[:, self.l_len // 2:, :]

        e_clickBasket_s3 = e_clickBasket[:, self.l_len - 2:, :]
        e_favorBasket_s3 = e_favorBasket[:, self.l_len - 2:, :]
        e_cartBasket_s3 = e_cartBasket[:, self.l_len - 2:, :]
        e_buyBasket_s3 = e_buyBasket[:, self.l_len - 2:, :]

        # EMA
        e_clickBasket_ema1 = self.EMA(e_clickBasket_s1, clk_dweight1, clk_dweight2)  #
        e_favorBasket_ema1 = self.EMA(e_favorBasket_s1, fav_dweight1, fav_dweight2)  #
        e_cartBasket_ema1 = self.EMA(e_cartBasket_s1, cart_dweight1, cart_dweight2)  #
        e_buyBasket_ema1 = self.EMA(e_buyBasket_s1, buy_dweight1, buy_dweight2)  #

        e_clickBasket_ema2 = self.EMA(e_clickBasket_s2, clk_dweight1, clk_dweight2)  #
        e_favorBasket_ema2 = self.EMA(e_favorBasket_s2, fav_dweight1, fav_dweight2)  #
        e_cartBasket_ema2 = self.EMA(e_cartBasket_s2, cart_dweight1, cart_dweight2)  #
        e_buyBasket_ema2 = self.EMA(e_buyBasket_s2, buy_dweight1, buy_dweight2)  #

        e_clickBasket_ema3 = self.EMA(e_clickBasket_s3, clk_dweight1, clk_dweight2)  #
        e_favorBasket_ema3 = self.EMA(e_favorBasket_s3, fav_dweight1, fav_dweight2)  #
        e_cartBasket_ema3 = self.EMA(e_cartBasket_s3, cart_dweight1, cart_dweight2)  #
        e_buyBasket_ema3 = self.EMA(e_buyBasket_s3, buy_dweight1, buy_dweight2)  #

        clk_s1 = e_clickBasket_ema1[:, -1, :]
        fav_s1 = e_favorBasket_ema1[:, -1, :]
        cart_s1 = e_cartBasket_ema1[:, -1, :]
        buy_s1 = e_buyBasket_ema1[:, -1, :]

        clk_s2 = e_clickBasket_ema2[:, -1, :]
        fav_s2 = e_favorBasket_ema2[:, -1, :]
        cart_s2 = e_cartBasket_ema2[:, -1, :]
        buy_s2 = e_buyBasket_ema2[:, -1, :]

        clk_s3 = e_clickBasket_ema3[:, -1, :]
        fav_s3 = e_favorBasket_ema3[:, -1, :]
        cart_s3 = e_cartBasket_ema3[:, -1, :]
        buy_s3 = e_buyBasket_ema3[:, -1, :]

        # multi-scale
        final1_ = clk_s1 + self.alpha[0] * (clk_s1 - clk_s2) + self.alpha[1] * (clk_s1 - clk_s3)
        final2_ = fav_s1 + self.alpha[0] * (fav_s1 - fav_s2) + self.alpha[1] * (fav_s1 - fav_s3)
        final3_ = cart_s1 + self.alpha[0] * (cart_s1 - cart_s2) + self.alpha[1] * (cart_s1 - cart_s3)
        final4_ = buy_s1 + self.alpha[0] * (buy_s1 - buy_s2) + self.alpha[1] * (buy_s1 - buy_s3)

        # long- and short-term fusing
        all_1_ = torch.cat([final1_, clk_l], dim=-1)
        alpha1_ = self.gate1(all_1_)
        alpha1_ = torch.sigmoid(alpha1_)
        final1 = alpha1_ * final1_ + (1 - alpha1_) * clk_l

        all_2_ = torch.cat([final2_, fav_l], dim=-1)
        alpha2_ = self.gate1(all_2_)
        alpha2_ = torch.sigmoid(alpha2_)
        final2 = alpha2_ * final2_ + (1 - alpha2_) * fav_l

        all_3_ = torch.cat([final3_, cart_l], dim=-1)
        alpha3_ = self.gate1(all_3_)
        alpha3_ = torch.sigmoid(alpha3_)
        final3 = alpha3_ * final3_ + (1 - alpha3_) * cart_l

        all_4_ = torch.cat([final4_, buy_l], dim=-1)
        alpha4_ = self.gate1(all_4_)
        alpha4_ = torch.sigmoid(alpha4_)
        final4 = alpha4_ * final4_ + (1 - alpha4_) * buy_l

        # multi-behavior fusing
        w = torch.softmax(self.w, dim=-1)
        s1 = w[0, 0] * clk_s1 + w[0, 1] * fav_s1 + w[0, 2] * cart_s1 + w[0, 3] * buy_s1
        s2 = w[1, 0] * clk_s2 + w[1, 1] * fav_s2 + w[1, 2] * cart_s2 + w[1, 3] * buy_s2
        s3 = w[2, 0] * clk_s3 + w[2, 1] * fav_s3 + w[2, 2] * cart_s3 + w[2, 3] * buy_s3
        l = w[3, 0] * clk_l + w[3, 1] * fav_l + w[3, 2] * cart_l + w[3, 3] * buy_l

        final_ = s1 + self.alpha[0] * (s1 - s2) + self.alpha[1] * (s1 - s3)
        all_ = torch.cat([final_, l], dim=-1)
        alpha_ = self.gate1(all_)
        alpha_ = torch.sigmoid(alpha_)
        final = alpha_ * final_ + (1 - alpha_) * l

        # prediction
        all_items_index = list(range(1, self.NB_ITEMS + 1))
        e_all_items = self.item_emb(Variable(torch.LongTensor(all_items_index).to(self.device)))
        score1 = torch.mm(final1, e_all_items.transpose(-1, -2))
        score2 = torch.mm(final2, e_all_items.transpose(-1, -2))
        score3 = torch.mm(final3, e_all_items.transpose(-1, -2))
        score4 = torch.mm(final4, e_all_items.transpose(-1, -2))
        score5 = torch.mm(final, e_all_items.transpose(-1, -2))
        #
        oldItems1_oneHot = y_onehot(oldItemsData[0], self.NB_ITEMS, self.device)
        oldItems2_oneHot = y_onehot(oldItemsData[1], self.NB_ITEMS, self.device)
        oldItems3_oneHot = y_onehot(oldItemsData[2], self.NB_ITEMS, self.device)
        oldItems4_oneHot = y_onehot(oldItemsData[3], self.NB_ITEMS, self.device)
        newItems_oneHot = 1 - oldItems1_oneHot - oldItems2_oneHot - oldItems3_oneHot - oldItems4_oneHot
        score = torch.mul(score1, oldItems1_oneHot) + torch.mul(score2, oldItems2_oneHot) + \
                torch.mul(score3, oldItems3_oneHot) + torch.mul(score4, oldItems4_oneHot) + \
                torch.mul(score5, newItems_oneHot)

        u = self.u_(e_user)
        score = score + u

        sim = torch.mm(e_all_items, e_all_items.transpose(-1, -2))
        all_omiga = self.omiga(Variable(torch.LongTensor(all_items_index).to(self.device)))
        return score, sim, all_omiga  #

    def sumPooling(self, e_data):
        e_basket = torch.sum(e_data, dim=-2)
        return e_basket

    def tensorMask(self, y):
        y_sum = torch.sum(y, dim=-1)
        y_posmask = (y_sum != 0).type_as(y_sum)
        y_negmask = 1 - y_posmask
        return [y_posmask, y_negmask]

    def EMA(self, sequence, weight1, weight2):  #
        weight1 = torch.sigmoid(weight1)
        weight2 = torch.sigmoid(weight2)
        output = torch.empty(size=(sequence.size()[0], sequence.size()[1], sequence.size()[2])).to(self.device)
        output[:, 0, :] = sequence[:, 0, :]
        for t in range(1, sequence.size()[1]):
            output[:, t, :] = torch.mul(weight1, output[:, t - 1, :].clone()) + torch.mul(
                (1 - weight2 * weight1), sequence[:, t, :].clone())  #

        return output

    def Attention1(self, query, key, value, attn_mask):
        scale_factor = 1 / math.sqrt(query.size(-1))
        key = self.fai1(self.w1(key))
        attn_weight = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) * scale_factor
        attn_weight += (attn_mask[1] * -1e20).unsqueeze(1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value).squeeze(1)

    def Attention2(self, query, key, value, attn_mask):
        scale_factor = 1 / math.sqrt(query.size(-1))
        key = self.fai2(self.w2(key))
        attn_weight = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) * scale_factor
        attn_weight += (attn_mask[1] * -1e20).unsqueeze(1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value).squeeze(1)

    def Attention3(self, query, key, value, attn_mask):
        scale_factor = 1 / math.sqrt(query.size(-1))
        key = self.fai3(self.w3(key))
        attn_weight = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) * scale_factor
        attn_weight += (attn_mask[1] * -1e20).unsqueeze(1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value).squeeze(1)

    def Attention4(self, query, key, value, attn_mask):
        scale_factor = 1 / math.sqrt(query.size(-1))
        key = self.fai4(self.w4(key))
        attn_weight = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)) * scale_factor
        attn_weight += (attn_mask[1] * -1e20).unsqueeze(1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return torch.matmul(attn_weight, value).squeeze(1)
