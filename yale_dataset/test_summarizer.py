from summarizer import Summarizer
import json
import pandas as pd

yale_doc_dict = json.load(open('yale_doc_dict.json', 'r'))

subject=[]
course= []
title=[]
description=[]
transcript=[]

cnt=0
model = Summarizer()
for key in yale_doc_dict:
    if cnt%1==0:
        print(cnt)

    body = yale_doc_dict[key]['transcript']
    
    subject.append(yale_doc_dict[key]['info']['department'])
    course.append(yale_doc_dict[key]['info']['course'])
    title.append(yale_doc_dict[key]['info']['lecture'])
    description.append(yale_doc_dict[key]['description'])
    transcript.append(model(body, ratio=0.08))
    
    cnt+=1
df = pd.DataFrame.from_dict({"subject":subject, "course":course, "title":title,"description":description,"transcript":transcript}) 
df.to_csv("Yale_dataset_extractive.csv")