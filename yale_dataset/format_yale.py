import nltk
import glob
import json
import os
import pandas as pd
nltk.download('punkt')

fns = glob.glob('./yale_dataset/*/*/*transcript.txt')

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


# step2 Input format: subject, course, title, description, transcript