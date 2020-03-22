import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from convlab2.dst.trade.crosswoz.utils.config import *
from convlab2.dst.trade.crosswoz.trade import *

'''
    python demo.py
'''

# specify model path
args['path'] = 'model/TRADE-multiwozdst/HDD100BSZ4DR0.2ACC-0.3228'
model = CrossWOZTRADE(args['path'])


user_act = '你好 ， 我想 找家 人均 消费 在 100 - 150 元 的 餐馆 吃 驴 杂汤 这 道菜 ， 请 给 我 推荐 一家 餐馆 用餐 吧 。'
model.state['history'] = [['user', user_act]]
state = model.update(user_act)
print(state)