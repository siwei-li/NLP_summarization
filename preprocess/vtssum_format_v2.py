import json
import glob
import pickle
import pandas as pd
import os

def GenerateDatasetFromJson():
    '''
    return: a list of list of list, 
        first dimension is each document,
        second dimension is doc_list and label_list of each document,
        third dimension is number of sentences in a document
    '''
    fns = glob.glob('./train/*.json')
    dataset=[]
    for fn in fns:
        file=json.load(open(fn,'r'))
        title = file['title']
        para = file['segmentation']
        summa = file['summarization']

        doc_list=[]
        label_list=[]
        for one_key in summa:
            for sentence_dict in summa[one_key]['summarization_data']:
                doc_list.append(sentence_dict['sent'])
                label_list.append(sentence_dict['label'])
        dataset.append([doc_list, label_list])
    return dataset

def GenerateDatasetFromJson2(dir,doc_num, neg_multiplier):
    '''
    return: a list of list of list, 
        first dimension is each document,
        second dimension is doc_list and label_list of each document,
        third dimension is number of sentences in a document
    '''
    fns = glob.glob(os.path.join(dir, '*.json'))
    cnt=1
    sent_list=[]
    doc_list = []
    label_list=[]
    for fn in fns:
        file=json.load(open(fn,'r'))
        title = file['title']
        para = ' '.join(map(lambda x: ' '.join(x), file['segmentation']))
        summa = file['summarization']

        for one_key in summa:
            for sentence_dict in summa[one_key]['summarization_data']:
                sent_list.append(sentence_dict['sent'])
                doc_list.append(para)
                label_list.append(sentence_dict['label'])
        cnt+=1
        if cnt>doc_num:
            break
    df = pd.DataFrame.from_dict({"sents":sent_list, "docs":doc_list, "y":label_list}) 
    pos_df = df[df.y == 1]
    neg_df = df[df.y == 0]

    print("Negative sample size:", len(neg_df))
    print("Positive sample size:", len(pos_df))

    sub_neg_df = neg_df.sample(len(pos_df)*neg_multiplier) 
    balanced_df = pos_df.append(sub_neg_df)
    return balanced_df

if __name__=="__main__":
    # dataset = GenerateDatasetFromJson()
    # pickle.dump(dataset, open('train_data.robin','wb'))
    # dataset = pickle.load(open('train_data.robin','rb'))
    # print(len(dataset))

    dataset_df = GenerateDatasetFromJson2('./train', 500, 2)
    dataset_df.to_json("train_data.json")
    dataset_df = GenerateDatasetFromJson2('./test', 100, 2)
    dataset_df.to_json("test_data.json")

