import config
import os
import utils
import nltk
nltk.download('wordnet')


out_file = open("../output/20220616_225927/test_outputs.txt",
                encoding='utf-8', mode='r')
print("data read successflly!")

total_references = []
total_candidates = []
wflag = False  
whq = 2


for line in out_file:
    if "Reference:" in line:
        wflag = True
    if "Candidate:" in line:
        wflag = True
    if wflag == True:
        k = str(line)
        l = len(k)
        tempstr = k[11:l - 1]
        if k[:11] == "Reference: ":
            total_references.append(tempstr.split())
        if k[:11] == "Candidate: ":
            total_candidates.append(tempstr.split())
        wflag = False


s_blue_score, meteor_score = utils.measure(references=total_references, candidates=total_candidates)
c_bleu = utils.corpus_bleu_score(references=total_references, candidates=total_candidates)

avg_scores = {'c_bleu': c_bleu, 's_bleu': s_blue_score, 'meteor': meteor_score}
print(avg_scores)

