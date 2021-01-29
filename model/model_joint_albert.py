import torch
import torch.nn as nn
from transformers.modeling_albert import ALBERT_PRETAINED_MODEL_ARCHIVE_MAP,\
    AlbertPreTainedModel,AlbertModel,AlbertConfig
from modul import IntentClassifier,SlotClassifier



class JointAlbert(AlbertPreTainedModel):
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "albert"

    def __init__(self,config, args, intent_label_lst,slot_label_lst):
        super(JointAlbert,self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.albert = AlbertModel(config=config)

        self.intent_classifier = IntentClassifier(config.hidden_size,self.num_intent_labels,args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size,self.num_slot_labels,args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags = self.num_slot_labels,batch_first=True)

    def forward(self,input_ids,attention_mask,token_type_ids,intent_label_ids,slot_labels_ids):
        outputs = self.albert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1] #[cls]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. intent softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1),intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1,self.num_intent_labels),intent_label_ids.view(-1))
            total_loss += intent_loss
        # 2. slot softmax
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1,self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits,active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1,self.num_slot_labels),slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits,slot_logits),) + outputs[1:]
        outputs = (total_loss,) + outputs

        return outputs









