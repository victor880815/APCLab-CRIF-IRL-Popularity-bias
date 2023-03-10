import os

from utils.utils import set_random_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import datetime
import numpy as np
from tqdm import tqdm
from utils.LogPrint import Logger
from convhis.ConvHis import ConvHis
from convhis.ConvHisConfig import ConvHisConfig
from agents.AgentEAR import AgentEAR
from agents.AgentEARConfig import AgentEARConfig
from recommendersystem.recsys import recsys
from user.UserSim import UserSim
from user.UserSimConfig import UserSimConfig
from dialoguemanager.DialogueManager import DialogueManager
from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig
from utils.config_yelp import ConfigYelp
from utils.config_last_fm import ConfigLastFM
from utils.config_last_fm_star import ConfigLastFMStar
from utils.config_yelp_star import ConfigYelpStar
from utils import user_att_count, user_att_count_not_subset, user_att_count_not_subset_yelp
from utils import user_att_count_yelp
from utils import utils
from utils.global_variable import *

import pandas as pd
from itertools import chain

eps = np.finfo(np.float32).eps.item()
config = None


def standard_reward(a):
    return (a - np.mean(a)) / (np.std(a) + eps)


def PG_train(agent, load_model_type, dm):
    use_gpu = True
    env = dm
    PG_lr = 0.001
    PG_discount_rate = 0.7
    data_name = args.data_name
    if data_name == YELP_STAR:
        data_name = YELP
    PG_train_data_path = "./data/" + data_name + '/' + UI_INTERACTION_DATA + '/review_dict_valid.json'

    PG_test_data_path = "./data/" + data_name + '/' + UI_INTERACTION_DATA + '/review_dict_test.json'
    PG_epoch_num = 30
    PG_optimizer = optim.SGD(agent.DPN.parameters(), lr=PG_lr, )
    RW_optimizer = optim.SGD(agent.RWN.parameters(), lr=PG_lr, )
    PG_save_step = 1
    RW_softmax = nn.Softmax(dim=-1)
    bce_loss = nn.BCELoss()
    ranking_loss = nn.MarginRankingLoss(margin=0.)

    with open(PG_train_data_path, "rb") as f:
        PG_train_data_list = json.load(f)
    with open(PG_test_data_path, "rb") as f:
        PG_test_data_list = json.load(f)

    date_str = datetime.date.today().isoformat()
    sys.stdout = Logger("PG-agentear-{}-{}-lr-{}.log" \
                        .format(data_name,date_str, str(PG_lr)))

    ui_train_list = []
    for user_str, items in PG_train_data_list.items():
        user_id = int(user_str)
        for item_id in items:
            ui_train_list.append([user_id, item_id])
    each_epoch_len = len(ui_train_list) // 10

    ui_test_list = []
    for user_str, items in PG_test_data_list.items():
        user_id = int(user_str)
        for item_id in items:
            ui_test_list.append([user_id, item_id])
    ui_test_array = np.array(ui_test_list)
    np.random.shuffle(ui_test_array)

    test_size = 4000
    if args.data_name in [LAST_FM_STAR, LAST_FM]:
        test_size = 4000     # Only do 4000 iteration for the sake of time
    if args.data_name in [YELP_STAR, YELP]:
        test_size = 2500     # Only do 2500 iteration for the sake of time

    agent.set_env(env)
    if load_model_type == "PG":
        print("load PG model ...")
        agent.load_model()
    else:
        print("no pretrian model...")

    time_str = datetime.datetime.now().isoformat()
    print("{} start PG ...".format(time_str))
    print("lr: {:g}".format(PG_lr))

    best_average_reward = 0.
    best_average_turn = 100.
    best_success_rate = 0.
    best_count = 0
#     #============================================================================================
#     f= open("./data/" + 'lastfm' + '/Graph_generate_data' + "/item_dict.json")
#     content = json.load(f)
    
#     pop_item_list = []
#     unpop_item_list = []
#     for i in range(len(content)):
#         if 2 in content[str(i)]['feature_index'] or 10 in content[str(i)]['feature_index']:
#             # print(content[str(i)]['feature_index'])
#             pop_item_list.append(i)
#         if 23 in content[str(i)]['feature_index'] or 12 in content[str(i)]['feature_index'] \
#         or 16 in content[str(i)]['feature_index'] or 6 in content[str(i)]['feature_index']  \
#         or 18 in content[str(i)]['feature_index'] or 11 in content[str(i)]['feature_index'] \
#         or 25 in content[str(i)]['feature_index'] or 14 in content[str(i)]['feature_index'] \
#         or 8 in content[str(i)]['feature_index'] or 7 in content[str(i)]['feature_index'] \
#         or 29 in content[str(i)]['feature_index'] or 27 in content[str(i)]['feature_index'] \
#         or 32 in content[str(i)]['feature_index'] or 31 in content[str(i)]['feature_index'] \
#         or 30 in content[str(i)]['feature_index'] :
#             unpop_item_list.append(i)
            
#     same_list = list(set(pop_item_list)&set(unpop_item_list))
#     for num in same_list:
#         if num in pop_item_list:
#             pop_item_list.remove(num)
#         if num in unpop_item_list:
#             unpop_item_list.remove(num)

#     print(len(pop_item_list))
#     print(len(unpop_item_list))
    
#     f.close()
#     #============================================================================================
    f= open("./data/" + 'lastfm' + '/Graph_generate_data' + "/user_item.json")
    content = json.load(f)
    
    item_list = []
    for i in range(len(content)):
        item = content[str(i)]
        item_list.append(item)
    item_list = list(chain.from_iterable(item_list))
    df = pd.DataFrame(data={'item':item_list})
    df_pop = pd.DataFrame()
    df_pop.index = (df.item.value_counts() / len(df)).index
    df_pop['percentage'] = df.item.value_counts() / len(df)
    
    count = 0
    pop_item_list = []
    unpop_item_list = []
    for i in range(len(df_pop)):
        count += float(df_pop.iloc[i])
        if count <= 0.2:
            pop_item_list.append(df_pop.index[i])
        if count >= 0.8:
            unpop_item_list.append(df_pop.index[i])
            
    print(count)
    print(len(pop_item_list))
    print(len(unpop_item_list))
    
    f.close()
    #============================================================================================
    for _ in range(PG_epoch_num):
        print("epoch: ", _)
        epoch_reward_sum = 0.
        epoch_turn_sum = 0.
        epoch_success_sum = 0.

        train_count = 0
        pop_turn_sum = 0
        unpop_turn_sum = 0
        for e_data in tqdm(random.sample(ui_train_list, 10000), ncols=0):
            action_pool, reward_pool, success, reward_train_list, pcu = agent.PG_train_one_episode(e_data[0], e_data[1], pop_item_list, unpop_item_list)

            epoch_reward_sum += sum(reward_pool)
            epoch_turn_sum += len(reward_pool)
            epoch_success_sum += success
            
            print('len(reward_pool)', len(reward_pool))
            if pcu == 1:
                pop_turn_sum += len(reward_pool)
                
            if pcu == -1:
                unpop_turn_sum += len(reward_pool)
            
            train_count += 1

            total_reward = 0.
            for index in reversed(range(len(reward_pool))):
                total_reward = total_reward * PG_discount_rate + reward_pool[index]
                reward_pool[index] = total_reward

            reward_pool = np.array(reward_pool)
            reward_pool = reward_pool.astype(float)
            reward_pool = standard_reward(reward_pool) # ?

            reward_pool_tensor = torch.from_numpy(reward_pool)
            action_pool_tensor = torch.stack(action_pool, 0)

            if use_gpu:
                reward_pool_tensor = reward_pool_tensor.cuda()
                action_pool_tensor = action_pool_tensor.cuda()

            RW_loss = torch.tensor(0.).cuda()
            for reward_tuple in reward_train_list:
                ask_reward, rec_reward, label_tensor = reward_tuple

                reward_tensor = torch.stack([ask_reward, rec_reward], 0)
                reward_tensor = RW_softmax(reward_tensor)

                RW_loss += bce_loss(reward_tensor, label_tensor.float().cuda())

            RW_optimizer.zero_grad()
            RW_loss.backward()
            RW_optimizer.step()

            loss = torch.sum(torch.mul(action_pool_tensor, reward_pool_tensor).mul(-1))
            PG_optimizer.zero_grad()
            loss.backward()
            PG_optimizer.step()
            
            if train_count % 1000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}:train epoch {}, reward {:g}, turn {:g}, success {:g}".
                      format(time_str, _, epoch_reward_sum / train_count,
                             epoch_turn_sum / train_count, epoch_success_sum / train_count))
        print('unpop_turn_sum', unpop_turn_sum)
        print('pop_turn_sum', pop_turn_sum)
        pcu = (1 / 15) * (((1 / len(unpop_item_list)) * unpop_turn_sum) - ((1 / len(pop_item_list)) * pop_turn_sum))
                
        time_str = datetime.datetime.now().isoformat()
        print("{}:train epoch {}, reward {:g}, turn {:g}, success {:g}, pcu {:g}".
              format(time_str, _, epoch_reward_sum / train_count,
                     epoch_turn_sum / train_count, epoch_success_sum / train_count, pcu))

        if (_ + 1) % PG_save_step == 0:
            sum_reward = 0.
            sum_turn = 0
            sum_success = 0
            episode_num = 0
            for e_data in tqdm(random.sample(ui_test_list, test_size), ncols=0):
                reward, turn, success, no_use = agent.PG_eva_one_episode(e_data[0], e_data[1], pop_item_list, unpop_item_list)
                episode_num += 1
                sum_reward += reward
                sum_turn += turn
                sum_success += success

            average_reward = float(sum_reward) / episode_num
            average_turn = float(sum_turn) / episode_num
            success_rate = float(sum_success) / episode_num
            if average_turn < best_average_turn:
                best_average_reward = average_reward
                best_average_turn = average_turn
                best_success_rate = success_rate
                agent.save_model(False)
                best_count = 0
            # if success_rate > best_success_rate:
            #     best_average_reward = average_reward
            #     best_average_turn = average_turn
            #     best_success_rate = success_rate
            #     agent.save_model(False)
            #     best_count = 0
            else:
                best_count += 1
                agent.save_model(False)

            time_str = datetime.datetime.now().isoformat()
            print("{}: test epoch {}, average_reward {:g}, average_turn {:g}, success_rate {:g}, \
                    best_average_reward: {:g}, best_average_turn {:g}, best_success_rate {:g}" \
                  .format(time_str, _, average_reward, average_turn, success_rate, \
                          best_average_reward, best_average_turn, best_success_rate))

            if best_count == 10:
                break


parser = argparse.ArgumentParser(description='train ear agent')
parser.add_argument('--mode', type=str,
                    help='choose from pretrain or PG')
parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                    help='One of {lastfm, lastfm_star, yelp, yelp_star}.')
args = parser.parse_args()


if args.data_name == LAST_FM:
    config = ConfigLastFM()
    user_att_count_not_subset.set_config(config)
    model_name = "iter299-2023-01-05" # lastfm
elif args.data_name == YELP:
    config = ConfigYelp()
    user_att_count_not_subset.set_config(config)
    model_name = "iter800-2022-09-06" # yelp 2loss
elif args.data_name == LAST_FM_STAR:
    config = ConfigLastFMStar()
    user_att_count_not_subset.set_config(config)
    model_name = "iter4260-2022-09-01" # lastfm_star
elif args.data_name == YELP_STAR:
    config = ConfigYelpStar()
    user_att_count_not_subset.set_config(config)
    model_name = "iter800-2022-09-03" # yelp* 3loss not equal

ch = ConvHis(ConvHisConfig(config))
agent = AgentEAR(AgentEARConfig(config), ch)
rec = recsys(config, convhis=ch)
ch.set_rec(rec)
rec.load_model(model_name, False)
rec.init_eval(config.adj_index)
usersim = UserSim(UserSimConfig(config))
dm = DialogueManager(DialogueManagerConfig(config), rec, agent, usersim, ch)
set_random_seed(1)
PG_train(agent, args.mode, dm)


# import os

# from utils.utils import set_random_seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import argparse
# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import json
# import random
# import datetime
# import numpy as np
# from tqdm import tqdm
# from utils.LogPrint import Logger
# from convhis.ConvHis import ConvHis
# from convhis.ConvHisConfig import ConvHisConfig
# from agents.AgentEAR import AgentEAR
# from agents.AgentEARConfig import AgentEARConfig
# from recommendersystem.recsys import recsys
# from user.UserSim import UserSim
# from user.UserSimConfig import UserSimConfig
# from dialoguemanager.DialogueManager import DialogueManager
# from dialoguemanager.DialogueManagerConfig import DialogueManagerConfig
# from utils.config_yelp import ConfigYelp
# from utils.config_last_fm import ConfigLastFM
# from utils.config_last_fm_star import ConfigLastFMStar
# from utils.config_yelp_star import ConfigYelpStar
# from utils import user_att_count, user_att_count_not_subset, user_att_count_not_subset_yelp
# from utils import user_att_count_yelp
# from utils import utils
# from utils.global_variable import *

# from torch.autograd import Variable

# eps = np.finfo(np.float32).eps.item()
# config = None


# def standard_reward(a):
#     return (a - np.mean(a)) / (np.std(a) + eps)


# def PG_train(agent, load_model_type, dm):
#     use_gpu = True
#     env = dm
#     PG_lr = 0.001
#     PG_discount_rate = 0.7
#     data_name = args.data_name
#     if data_name == YELP_STAR:
#         data_name = YELP
#     PG_train_data_path = "./data/" + data_name + '/' + UI_INTERACTION_DATA + '/review_dict_valid.json'

#     PG_test_data_path = "./data/" + data_name + '/' + UI_INTERACTION_DATA + '/review_dict_test.json'
#     PG_epoch_num = 5
#     PG_optimizer = optim.SGD(agent.DPN.parameters(), lr=PG_lr, )
#     RW_optimizer = optim.SGD(agent.RWN.parameters(), lr=PG_lr, )
#     PG_save_step = 1
#     RW_softmax = nn.Softmax(dim=-1)
#     bce_loss = nn.BCELoss()
#     ranking_loss = nn.MarginRankingLoss(margin=0.)

#     with open(PG_train_data_path, "rb") as f:
#         PG_train_data_list = json.load(f)
#     with open(PG_test_data_path, "rb") as f:
#         PG_test_data_list = json.load(f)

#     date_str = datetime.date.today().isoformat()
#     sys.stdout = Logger("PG-agentear-{}-{}-lr-{}.log" \
#                         .format(data_name,date_str, str(PG_lr)))

#     # print('PG_train_data_list.items()PG_train_data_list', len(PG_train_data_list))
#     ui_train_list = []
#     for user_str, items in PG_train_data_list.items(): #user????????????item
#         user_id = int(user_str)
#         # print(user_id)
#         # print(items)
#         for item_id in items:
#             ui_train_list.append([user_id, item_id])
#     each_epoch_len = len(ui_train_list) // 10

#     ui_test_list = []
#     for user_str, items in PG_test_data_list.items():
#         user_id = int(user_str)
#         for item_id in items:
#             ui_test_list.append([user_id, item_id])
#     ui_test_array = np.array(ui_test_list)
#     np.random.shuffle(ui_test_array)

#     test_size = 4000
#     if args.data_name in [LAST_FM_STAR, LAST_FM]:
#         test_size = 4000     # Only do 4000 iteration for the sake of time
#     if args.data_name in [YELP_STAR, YELP]:
#         test_size = 2500     # Only do 2500 iteration for the sake of time

#     agent.set_env(env)
#     if load_model_type == "PG":
#         print("load PG model ...")
#         agent.load_model()
#     else:
#         print("no pretrian model...")

#     time_str = datetime.datetime.now().isoformat()
#     print("{} start PG ...".format(time_str))
#     print("lr: {:g}".format(PG_lr))

#     best_average_reward = 0.
#     best_average_turn = 100.
#     best_success_rate = 0.
#     best_count = 0

#     for _ in range(PG_epoch_num):
#         print("epoch: ", _)
#         epoch_reward_sum = 0.
#         epoch_turn_sum = 0.
#         epoch_success_sum = 0.

#         train_count = 0
#         for e_data in tqdm(random.sample(ui_train_list, 10000), ncols=0):
#             # print(e_data[0], e_data[1])
#             action_pool, reward_pool, success, reward_train_list = agent.PG_train_one_episode(e_data[0], e_data[1])
#             # print('action_poolaction_pool', action_pool)
#             epoch_reward_sum += sum(reward_pool)
#             epoch_turn_sum += len(reward_pool)
#             epoch_success_sum += success

#             train_count += 1

#             total_reward = 0.
#             for index in reversed(range(len(reward_pool))):
#                 total_reward = total_reward * PG_discount_rate + reward_pool[index]
#                 reward_pool[index] = total_reward

#             reward_pool = np.array(reward_pool)
#             reward_pool = reward_pool.astype(float)
#             reward_pool = standard_reward(reward_pool) # ?

#             reward_pool_tensor = torch.from_numpy(reward_pool)
#             action_pool_tensor = torch.stack(action_pool, 0)
            
#             if use_gpu:
#                 reward_pool_tensor = reward_pool_tensor.cuda()
#                 action_pool_tensor = action_pool_tensor.cuda()
                
#             RW_loss = torch.tensor(0.).cuda()
#             label_tensor_list = []
#             for reward_tuple in reward_train_list:
#                 ask_reward, rec_reward, label_tensor = reward_tuple
#                 label_tensor_list.append(label_tensor[0])
#                 reward_tensor = torch.stack([ask_reward, rec_reward], 0)
#                 reward_tensor = RW_softmax(reward_tensor)
#                 RW_loss += bce_loss(reward_tensor, label_tensor.float().cuda())

#             RW_optimizer.zero_grad()
#             RW_loss.backward()
#             RW_optimizer.step()
            
#             label_tensor_list = np.array(label_tensor_list)
#             label_tensor_list = label_tensor_list.astype(float)
#             label_tensor_list_tensor = torch.from_numpy(label_tensor_list)
#             label_tensor_list_tensor = label_tensor_list_tensor.cuda()
#             label_tensor_list_tensor = Variable(label_tensor_list_tensor, requires_grad=True)
            
#             # loss = torch.sum(torch.mul(action_pool_tensor, reward_pool_tensor).mul(-1))
#             loss = torch.sum(torch.mul(label_tensor_list_tensor, reward_pool_tensor).mul(-1))
            
#             PG_optimizer.zero_grad()
#             loss.backward()
#             PG_optimizer.step()

#             if train_count % 1000 == 0:
#                 time_str = datetime.datetime.now().isoformat()
#                 print("{}:train epoch {}, reward {:g}, turn {:g}, success {:g}".
#                       format(time_str, _, epoch_reward_sum / train_count,
#                              epoch_turn_sum / train_count, epoch_success_sum / train_count))

#         time_str = datetime.datetime.now().isoformat()
#         print("{}:train epoch {}, reward {:g}, turn {:g}, success {:g}".
#               format(time_str, _, epoch_reward_sum / train_count,
#                      epoch_turn_sum / train_count, epoch_success_sum / train_count))

#         if (_ + 1) % PG_save_step == 0:
#             sum_reward = 0.
#             sum_turn = 0
#             sum_success = 0
#             episode_num = 0
#             for e_data in tqdm(random.sample(ui_test_list, test_size), ncols=0):
#                 reward, turn, success, no_use = agent.PG_eva_one_episode(e_data[0], e_data[1])
#                 episode_num += 1
#                 sum_reward += reward
#                 sum_turn += turn
#                 sum_success += success

#             average_reward = float(sum_reward) / episode_num
#             average_turn = float(sum_turn) / episode_num
#             success_rate = float(sum_success) / episode_num
#             if average_turn < best_average_turn:
#                 best_average_reward = average_reward
#                 best_average_turn = average_turn
#                 best_success_rate = success_rate
#                 agent.save_model(False)
#                 best_count = 0
#             # if success_rate > best_success_rate:
#             #     best_average_reward = average_reward
#             #     best_average_turn = average_turn
#             #     best_success_rate = success_rate
#             #     agent.save_model(False)
#             #     best_count = 0
#             else:
#                 best_count += 1
#                 agent.save_model(False)

#             time_str = datetime.datetime.now().isoformat()
#             print("{}: test epoch {}, average_reward {:g}, average_turn {:g}, success_rate {:g}, \
#                     best_average_reward: {:g}, best_average_turn {:g}, best_success_rate {:g}" \
#                   .format(time_str, _, average_reward, average_turn, success_rate, \
#                           best_average_reward, best_average_turn, best_success_rate))

#             if best_count == 10:
#                 break


# parser = argparse.ArgumentParser(description='train ear agent')
# parser.add_argument('--mode', type=str,
#                     help='choose from pretrain or PG')
# parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
#                     help='One of {lastfm, lastfm_star, yelp, yelp_star}.')
# args = parser.parse_args()


# if args.data_name == LAST_FM:
#     config = ConfigLastFM() #??????dataset config
#     user_att_count_not_subset.set_config(config) #??????dataset
#     model_name = "iter299-2022-12-11" # lastfm
# elif args.data_name == YELP:
#     config = ConfigYelp()
#     user_att_count_not_subset.set_config(config)
#     model_name = "iter800-2022-09-06" # yelp 2loss
# elif args.data_name == LAST_FM_STAR:
#     config = ConfigLastFMStar()
#     user_att_count_not_subset.set_config(config)
#     model_name = "iter4260-2022-09-01" # lastfm_star
# elif args.data_name == YELP_STAR:
#     config = ConfigYelpStar()
#     user_att_count_not_subset.set_config(config)
#     model_name = "iter800-2022-09-03" # yelp* 3loss not equal

# ch = ConvHis(ConvHisConfig(config))
# agent = AgentEAR(AgentEARConfig(config), ch)
# rec = recsys(config, convhis=ch)
# ch.set_rec(rec)
# rec.load_model(model_name, False)
# rec.init_eval(config.adj_index)
# usersim = UserSim(UserSimConfig(config)) #???????????????????????????????????????????????????????????????
# dm = DialogueManager(DialogueManagerConfig(config), rec, agent, usersim, ch)
# set_random_seed(1)
# PG_train(agent, args.mode, dm)
