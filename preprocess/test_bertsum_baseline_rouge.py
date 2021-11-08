from pyrouge import Rouge155
from summarizer import Summarizer
import glob

def evaluate_summ(gold, gen_sum):
    # print("-----------------------------------------------------")
    # print("Original summary")
    # print(gold)
    # print("-----------------------------------------------------")
    # print("Generated summary")
    # print(gen_sum)
    # print("-----------------------------------------------------")
    rouge = Rouge155()
    score = rouge.score_summary(gold, gen_sum)
    print("Rouge1 Score: ",score)
    return score
# evaluate_summ()

fns = glob.glob('./articles/*.txt')
avg_ratio = 0
cnt=0
model = Summarizer()

for i, fn in enumerate(fns):
    with open(fn, 'r', encoding="utf-8") as f:
        try:
            lines = f.readlines()       
            lines.sort(key=lambda x:len(x)) 
            body = lines[-1]
            gold_standard = lines[-2]

            this_ratio = (len(gold_standard)/len(body))  
            avg_ratio+=this_ratio
        except:
            pass

        gen_sum = model(body, ratio=0.05)  # Specified with ratio 
        # score = evaluate_summ(gold_standard, gen_sum)
        print(i, fn)
        print(gen_sum)