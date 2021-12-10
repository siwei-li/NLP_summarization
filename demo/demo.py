from typing import final
import nltk
import glob
import json
import os
import pandas as pd
nltk.download('punkt')

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # for debugging
import time
import pickle
import torch
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
import transformers
from torch.utils.data import Dataset, DataLoader 


from transformers import AutoTokenizer, AutoModel
sentenc_model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(sentenc_model_name)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from tqdm.notebook import tqdm

# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the BART modules from huggingface/transformers
from transformers import BartTokenizerFast, BartForConditionalGeneration
device = 'cuda' if cuda.is_available() else 'cpu'

def PrepareData():
    # fns = glob.glob('./yale_dataset/*/*/*transcript.txt')
    fns = ['./yale_dataset\\african-american-studies\\afam-162\\lecture-16_transcript.txt',
    './yale_dataset\\biomedical-engineering\\beng-100\\lecture-5_transcript.txt',
    './yale_dataset\\political-science\\plsc-114\\lecture-22_transcript.txt']

    # fns = ['./yale_dataset/african-american-studies/afam-162/lecture-1_transcript.txt']
    cnt = 0
    sent_list = []
    doc_index_list = []
    yale_doc_dict = {}

    for fn in fns:
        with open(fn ,'r', encoding='utf8') as f:
            para = f.readlines()
            if len(para)==0:
                continue
            para = para[0]
            a_list = nltk.tokenize.sent_tokenize(para)
            sent_list.extend(a_list)
            
            dirs = fn.split('\\')
            department_name = dirs[-3]
            course_name = dirs[-2]
            lecture_name = dirs[-1].split('_')[0]
            
            info = {'department':department_name, 'course':course_name, 'lecture':lecture_name}

            summa_fn = fn.split('_transcript.txt')[0]+'_overview.txt'
            
            with open(summa_fn, 'r', encoding='utf8') as f2:
                lines = f2.readlines()
                description = ''
                if len(lines)>=2:
                    description = lines[1]
                yale_doc_dict[cnt]={'info':info, 'description': description,'transcript':para}
        
        doc_index_list.extend([cnt]*len(a_list))
        cnt+=1

    df = pd.DataFrame.from_dict({"sents":sent_list, "docs":doc_index_list}) 
    df.to_json("yale_data.json")

    with open('yale_doc_dict.json', 'w') as json_file:
        json.dump(yale_doc_dict, json_file)

    return df, yale_doc_dict

# Create a Data Loader Class
class YaleData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, doc_dict, device):
        self.len = len(dataframe)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dict_list = []

        for index in range(dataframe.shape[0]):
            if index%1000==0:
                print(f"CNNDailyMailData: {index}/{dataframe.shape[0]}")

            sentence = dataframe.iloc[index].sents
            document = doc_dict[dataframe.iloc[index].docs]['transcript']

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
                'doc_mask': torch.tensor(mask[1], dtype=torch.long, device=device)
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
        self.pre_classifier = torch.nn.Linear(in_features*3, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
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



def ValidateModelYale(model, testing_loader, test_df, doc_dict):
    model.eval()

    subject=[]
    course= []
    title=[]
    description=[]
    transcript=[]

    summary = {}
    with torch.no_grad():
        for i, data in enumerate(testing_loader, 0): 
            outputs = model(data['sent_ids'], data['doc_ids'], data['sent_mask'], data['doc_mask']) 

            if torch.count_nonzero(outputs > 0.5)!=0:
                bias_arr = torch.where(outputs>0.5)[0].cpu().detach().numpy()
                for bias in bias_arr:
                    doc_id = test_df.loc[i*testing_loader.batch_size + bias, 'docs']
                    key = doc_id
                    if key not in summary:
                        summary[key] = []
                    
                    summary[key].append(test_df.loc[i*testing_loader.batch_size + bias, 'sents'])
                          
    for key in summary:
        subject.append(doc_dict[key]['info']['department'])
        course.append(doc_dict[key]['info']['course'])
        title.append(doc_dict[key]['info']['lecture'])
        description.append(doc_dict[key]['description'])
        transcript.append(' '.join((summary[key])))
    
    df = pd.DataFrame.from_dict({"subject":subject, "course":course, "title":title,"description":description,"transcript":transcript}) 

    return df
def ExtractiveModel(yale_df, yale_doc_dict):
    MAX_LEN = 512
    VALID_BATCH_SIZE = 4

    

    print( "Yale shape", yale_df.shape)

    yale_set = YaleData(yale_df, tokenizer, MAX_LEN, yale_doc_dict, device)


    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                    }

    yale_loader = DataLoader(yale_set, **test_params)

    model = SentenceBertClass(model_name=sentenc_model_name)
    model.to(device)

    model.load_state_dict(torch.load("models/minilm_bal_exsum.pth"))
    summary_df = ValidateModelYale(model, yale_loader, yale_df, yale_doc_dict)
    return summary_df
    





class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.description
        self.ctext = self.data.transcript

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt',truncation=True)
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt',truncation=True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def Predict(df):
    TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
    TRAIN_EPOCHS = 20        # number of epochs to train (default: 10)
    VAL_EPOCHS = 1 
    LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512
    SUMMARY_LEN = 150 

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    
    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    df = df[['transcript','description']]
    df = df[(df['transcript'].str.len()>200) & (df['description'].str.len()>20)]
    df.description = 'summarize: ' + df.description
    print(df.head())

    val_dataset=df

    # Creating the Training and Validation dataset for further creation of Dataloader
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }


    val_loader = DataLoader(val_set, **val_params)

    # Loading model

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.load_state_dict(torch.load("./models/abs_model"))
    model.to(device)

    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    
    predictions, actuals = validate(tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    
    return final_df

if __name__=="__main__":
    yale_df, yale_doc_dict = PrepareData()

    summary_df = ExtractiveModel(yale_df, yale_doc_dict)


    summary_df.to_csv("Yale_dataset_extractive_my.csv")

    final_df = Predict(summary_df)

    for index, row in final_df.iterrows():
        print(row['Generated Text'])
        print(row['Actual Text'])
        print('------------------------------------------------------------------------------------')
    final_df.to_csv("predictions_test.csv")