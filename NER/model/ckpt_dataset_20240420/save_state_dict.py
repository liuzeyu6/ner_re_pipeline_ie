import torch
import sys
sys.path.append("../../BERT_BiLSTM_CRF")
model = torch.load("ner_model_25.pth")
torch.save(model.state_dict(), "ner_model.pth")