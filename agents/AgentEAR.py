import torch
import datetime
from torch.distributions import Categorical
from agents.DeepPolicyNetwork import TwoLayersModel
from agents.RewardNetwork import RewardModel
from agents.RecommendFailed import RecommendFailed
from agents.RewardJudgeSingleAction import human_feedback
from torch.autograd import Variable

class AgentEAR:
    def __init__(self, config, convhis):
        self.convhis = convhis
        self.use_gpu = config.use_gpu
        self.DPN = TwoLayersModel(config)
        self.RWN = RewardModel(config)
        self.DPN_model_path = config.DPN_model_path
        self.DPN_model_name = config.DPN_model_name
        self.aciton_len = config.output_dim

        self.rec = None
        self.env = None

    def set_rec_model(self, rec_model):
        self.rec = rec_model

    def set_env(self, env):
        self.env = env

    def get_reward(self, state):
        return self.RWN(state.cuda())

    def init_episode(self):
        self.DPN.eval()
        self.RWN.eval()

    def save_model(self, is_preatrain):
        if is_preatrain:
            name_suffix = "_PRE"
        else:
            name_suffix = "_PG"
        time_str = datetime.datetime.now().isoformat()
        torch.save(self.DPN.state_dict(), "".join([self.DPN_model_path, self.DPN_model_name + name_suffix + time_str]))
        torch.save(self.RWN.state_dict(), "".join([self.DPN_model_path, self.DPN_model_name + "_RWN" + time_str]))

    def load_model(self):
        name_suffix = "_PG2022-12-18T04:46:22.216016"
        self.DPN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + name_suffix])))
        self.RWN.load_state_dict(torch.load("/".join([self.DPN_model_path, self.DPN_model_name + "_RWN2022-12-18T04:46:22.216016"])))

    #================================專家決策================================
    def PG_train_one_episode(self, user, item, pop_item_list, unpop_item_list):
        self.DPN.train()
        self.RWN.train()
        state_pool = []
        action_pool = []
        reward_pool = []
        reward_train_list = []

        state = self.env.initialize_episode(user, item)
        IsOver = False
        success = False
        IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
        feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, pcu = self.env.step(torch.tensor(0).cuda(), pop_item_list, unpop_item_list)
        while not IsOver:
            attribute_distribution = self.DPN(state.float().cuda(), True)
            c = Categorical(probs = attribute_distribution)

            action = c.sample()#透過DPN吐出action
            # print('actionactionaction', type(action))
            # print('actionaction', action)

            # IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
            # feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state = self.env.step(action) #放入action
            label_tensor = torch.tensor([0, 1])
            ask_reward_more = human_feedback(be_len, be_rank, ask_len, ask_rank, feedback_rec_len, feedback_rec_rank)
            if ask_reward_more:
                label_tensor = torch.tensor([1, 0])
            label_tensor_new = label_tensor[1].cuda()
            label_tensor_new = Variable(label_tensor_new.float(), requires_grad=True)
            
            IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
            feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, pcu = self.env.step(label_tensor_new, pop_item_list, unpop_item_list) #放入action
            # print(pcu)
            reward_train_list.append((ask_reward, feedback_rec_reward, label_tensor))

            state_pool.append(state)
            # print(c.log_prob(action))
            # print('123',label_tensor_new)
            # action_pool.append(c.log_prob(action))
            action_pool.append(label_tensor_new)
            reward_pool.append(reward.item())

            if not IsOver:
                state = next_state
        # print('pcu', pcu)
        return action_pool, reward_pool, success, reward_train_list, pcu

#     #================================原本沒改過的================================
#     def PG_train_one_episode(self, user, item, pop_item_list, unpop_item_list):
#         self.DPN.train()
#         self.RWN.train()
#         state_pool = []
#         action_pool = []
#         reward_pool = []
#         reward_train_list = []

#         state = self.env.initialize_episode(user, item)
#         IsOver = False
#         success = False
#         while not IsOver:
#             attribute_distribution = self.DPN(state.float().cuda(), True)
#             c = Categorical(probs = attribute_distribution)

#             action = c.sample()
            
#             IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
#             feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, pcu = self.env.step(action, pop_item_list, unpop_item_list)
#             label_tensor = torch.tensor([0, 1])
#             ask_reward_more = human_feedback(be_len, be_rank, ask_len, ask_rank, feedback_rec_len, feedback_rec_rank)
#             if ask_reward_more:
#                 label_tensor = torch.tensor([1, 0])
#             reward_train_list.append((ask_reward, feedback_rec_reward, label_tensor))

#             state_pool.append(state)
#             action_pool.append(c.log_prob(action))
#             reward_pool.append(reward.item())

#             if not IsOver:
#                 state = next_state

#         return action_pool, reward_pool, success, reward_train_list, pcu

#     #================================原本沒改過的================================
#     def PG_eva_one_episode(self, user, item, pop_item_list, unpop_item_list, silence=True):
#         self.DPN.eval()
#         self.RWN.eval()
#         total_reward = 0.
#         turn_count = 0
#         is_success = False
#         state_list = []

#         state = self.env.initialize_episode(user, item)
#         IsOver = False
#         while not IsOver:
#             turn_count += 1
#             attribute_distribution = self.DPN(state.float().cuda(), True)

#             action = int(attribute_distribution.argmax())

#             IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
#             feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, pcu = self.env.step(action, pop_item_list, unpop_item_list)
#             state_list.append(step_state)
#             total_reward += reward
#             is_success = success
#             if not IsOver:
#                 state = next_state

#         return total_reward, turn_count, is_success, state_list    

    #================================專家決策================================
    def PG_eva_one_episode(self, user, item, pop_item_list, unpop_item_list, silence=True):
        self.DPN.eval()
        self.RWN.eval()
        total_reward = 0.
        turn_count = 0
        is_success = False
        state_list = []

        state = self.env.initialize_episode(user, item)
        IsOver = False
        IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, pcu = self.env.step(0, pop_item_list, unpop_item_list)
        
        while not IsOver:
            turn_count += 1
            # attribute_distribution = self.DPN(state.float().cuda(), True)
            # action = int(attribute_distribution.argmax())
            
            label_tensor = torch.tensor([0, 1])
            ask_reward_more = human_feedback(be_len, be_rank, ask_len, ask_rank, feedback_rec_len, feedback_rec_rank)
            if ask_reward_more:
                label_tensor = torch.tensor([1, 0])
            label_tensor = int(label_tensor[1])

            # print('label_tensorlabel_tensor', type(label_tensor))
            # print('label_tensor', label_tensor)
            IsOver, next_state, reward, success, be_len, be_rank, ask_len, ask_rank, \
            feedback_rec_len, feedback_rec_rank, ask_reward, feedback_rec_reward, step_state, pcu = self.env.step(label_tensor, pop_item_list, unpop_item_list)
            
            
            state_list.append(step_state)
            total_reward += reward
            is_success = success
            if not IsOver:
                state = next_state
            
                
        return total_reward, turn_count, is_success, state_list 