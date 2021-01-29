import json
import os
from argparse import Namespace
import torch
from torch.utils.data import TensorDataset
from utils import InputExample,InputFeatures,load_tokenizer,JointSlotIntentDataProcess
from trainer import Trainer
from common.logger import logger
import codecs
import random
import copy



class JointBertModelPredictHandler(JointSlotIntentDataProcess):
    def __init__(self,config_file_name):
        config = json.load(open(os.path.join(config_file_name,'params_config.json'),'r',encoding='utf-8'))
        self.config = Namespace(**config)
        self.tokenizer = load_tokenizer(self.config)
        self.trainer = Trainer(self.config)
        self.slot_labels,self.intent_labels = self.config.slot_label_lst,self.config.intent_label_lst







