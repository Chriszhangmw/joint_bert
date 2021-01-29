

import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self,input_dim,num_intent_labels,dropout_rate=0.):
        super(IntentClassifier,self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.liner = nn.Linear(input_dim,num_intent_labels)
    def forward(self,x):
        x = self.dropout(x)
        return self.liner(x)

class SlotClassifier(nn.Module):
    def __init__(self,input_dim,num_intent_labels,dropout_rate=0.):
        super(SlotClassifier,self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.liner = nn.Linear(input_dim,num_intent_labels)

    def forward(self,x):
        x = self.dropout(x)
        return self.liner(x)









