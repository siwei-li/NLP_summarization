import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # for debugging
import time
import json
import pickle
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
import pandas as pd 
import transformers
from torch.utils.data import Dataset, DataLoader 


from transformers import AutoTokenizer, AutoModel
sentenc_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(sentenc_model_name)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from tqdm.notebook import tqdm

import os 


# Create a Data Loader Class
class CNNDailyMailData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, doc_dict, device):
        self.len = len(dataframe)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dict_list = []

        for index in range(dataframe.shape[0]):
            if index%1000==0:
                print(f"CNNDailyMailData: {index}/{dataframe.shape[0]}")

            sentence = dataframe.iloc[index].sents
            document = doc_dict[str(dataframe.iloc[index].docs)]

            inputs = self.tokenizer.batch_encode_plus(
                [sentence, document], 
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']

            mask = inputs['attention_mask']

            self.dict_list.append({
                'sent_ids': torch.tensor(ids[0], dtype=torch.long, device=device),
                'doc_ids': torch.tensor(ids[1], dtype=torch.long, device=device),
                'sent_mask': torch.tensor(mask[0], dtype=torch.long, device=device),
                'doc_mask': torch.tensor(mask[1], dtype=torch.long, device=device),
                'targets': torch.tensor([dataframe.iloc[index].y], dtype=torch.float, device=device)
            })

    def __getitem__(self, index):
        return self.dict_list[index]
    
    def __len__(self):
        return self.len

# get mean pooling for sentence bert models 
# ref https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
# Note that different sentence transformer models may have different in_feature sizes
class SentenceBertClass(torch.nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", in_features=384):
        super(SentenceBertClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(in_features*3, 768)       # 768 org, try 384
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)                       # 768 org
        self.classifierSigmoid = torch.nn.Sigmoid()

    def forward(self, sent_ids, doc_ids, sent_mask, doc_mask):

        sent_output = self.l1(input_ids=sent_ids, attention_mask=sent_mask) 
        sentence_embeddings = mean_pooling(sent_output, sent_mask) 

        doc_output = self.l1(input_ids=doc_ids, attention_mask=doc_mask) 
        doc_embeddings = mean_pooling(doc_output, doc_mask)

        # elementwise product of sentence embs and doc embs
        combined_features = sentence_embeddings * doc_embeddings  

        # Concatenate input features and their elementwise product
        concat_features = torch.cat((sentence_embeddings, doc_embeddings, combined_features), dim=1)   
        
        pooler = self.pre_classifier(concat_features) 
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.classifierSigmoid(output) 

        return output



def validate_model(model, testing_loader, test_df, print_n_steps):
    model.eval()

    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    recall_denominator = 0
    recall_numerator = 0
    last_time_stamp=time.time()

    with torch.no_grad():
        for i, data in enumerate(testing_loader, 0): 
            targets = data['targets']
            outputs = model(data['sent_ids'], data['doc_ids'], data['sent_mask'], data['doc_mask']) 
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            
            '''
            if torch.count_nonzero(outputs > 0.5)!=0:
                bias_arr = torch.where(outputs>0.5)[0].cpu().detach().numpy()
                for bias in bias_arr:
                    print(test_df.iloc[i*testing_loader.batch_size + bias].y)
                #i*4+bias
            '''
            n_correct += torch.count_nonzero(targets == (outputs > 0.5)).item()
            recall_denominator += torch.count_nonzero(targets == 1).item()
            recall_numerator += torch.count_nonzero((targets == 1)&(outputs>0.5)).item()

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if i%print_n_steps==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(str(i* test_params["batch_size"]) + "/" + str(len(test_df)) + " - Steps. Acc ->", accu_step, "Loss ->", loss_step, "Time ->",  time.time()-last_time_stamp)
                last_time_stamp = time.time()

             
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    print(f"Validation Recall Epoch: {recall_numerator}/{recall_denominator}")
    
    return epoch_accu

def train(model, training_loader, epoch, print_n_steps):    
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    last_time_stamp=time.time()

    for i,data in enumerate(training_loader, 0):
        targets = data['targets']
        outputs = model(data['sent_ids'], data['doc_ids'], data['sent_mask'], data['doc_mask']) 
        loss = loss_function(outputs, targets)
        tr_loss += loss.item() 
        n_correct += torch.count_nonzero(targets == (outputs > 0.5)).item()

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if i!=0 and i%print_n_steps==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            # print(str(i* train_params["batch_size"]) + "/" + str(len(train_df)) + " - Steps. Acc ->", accu_step, "Loss ->", loss_step, "Time ->",  time.time()-last_time_stamp)
            acc_step_holder.append(accu_step), loss_step_holder.append(loss_step)
            last_time_stamp = time.time()
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    # print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return model


if __name__=="__main__":
    sum_dir = "" # location to store and load models

    # Defining some key variables that will be used later on in the training
    TRAIN_FROM_SCRATCH = True
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-05 
    device = 'cuda' if cuda.is_available() else 'cpu'

    # load dataframes containining preprocessed samples from CNN/Dailymail Dataset
    train_df = pd.read_json(sum_dir + "train_data.json")

    train_doc_dict = json.load(open('train_doc_dict.json', 'r'))
    test_df = pd.read_json(sum_dir +"test_data.json") 
    test_doc_dict = json.load(open('test_doc_dict.json', 'r'))
    print( "Train, test shape", train_df.shape, test_df.shape)

    t1=time.time()
    if os.path.isfile('training_set.robin') and os.path.isfile('testing_set.robin'):
        training_set = pickle.load(open('training_set.robin', 'rb'))
        testing_set = pickle.load(open('testing_set.robin', 'rb'))
    else:
        training_set = CNNDailyMailData(train_df, tokenizer, MAX_LEN, train_doc_dict, device)
        testing_set = CNNDailyMailData(test_df, tokenizer, MAX_LEN, test_doc_dict, device)
        pickle.dump(training_set, open('training_set.robin','wb'))
        pickle.dump(testing_set, open('testing_set.robin','wb'))

    t2=time.time()
    print("Preprocess time: ", t2-t1)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    weight_decay_list = [(9, 0)]

    for number, weight_decay in weight_decay_list:
        model = SentenceBertClass(model_name=sentenc_model_name)
        model.to(device)
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
        print_n_steps = 300
        if TRAIN_FROM_SCRATCH:
            # if os.path.isfile('models/minilm_bal_exsum.pth'):
            #     print("Loaded model from models/minilm_bal_exsum.pth")
            #     model.load_state_dict(torch.load("models/minilm_bal_exsum.pth"))
            # Defining the training function on the 80% of the dataset for tuning the distilbert model
            
            acc_step_holder, loss_step_holder = [], []

            for epoch in range(EPOCHS):
                model = train(model, training_loader, epoch, print_n_steps)
                torch.save(model.state_dict(), f"models/minilm_bal_exsum{number}.pth")

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5))
            ax1.plot(acc_step_holder, label="Accuracy")
            ax2.plot(loss_step_holder, label="Loss")
            ax1.title.set_text("Accuracy")
            ax2.title.set_text("Loss")
            fig.tight_layout()
            # plt.show()
            plt.savefig(f'train{number}.png')

            acc = validate_model(model, testing_loader, test_df, print_n_steps)
            print(f"{number}Accuracy on test data = %0.2f%%" % acc)


            """Hint: Try a larger sentence embedding [pretrained model](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models) to improve overall train/test accuracy.
            """

            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/minilm_bal_exsum.pth")

        else:
            model.load_state_dict(torch.load("models/minilm_bal_exsum.pth"))
            acc = validate_model(model, testing_loader, test_df, print_n_steps)
            print("Accuracy on test data = %0.2f%%" % acc)
