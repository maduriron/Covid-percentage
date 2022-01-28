import json
import pandas as pd

json_1 = "./dicts/d_resnest_softargmax&regr&hp_1_05_finalcheck.json" # checked (inference 4_hp) (test)
json_2 = "./dicts/d_resnext_softargmax&regr_1_05_finalcheck.json" # checked (inference 4) (test)
json_3 = "./dicts/d_resnest_KL&exp&regr&hp_1_05_finalcheck.json" # checked (inference 7_hp) (test)
json_4 = "./dicts/d_resnest101_softargmax&regr&hp_1_05_finalcheck.json" # checked (inference 4_hp) (test)


d_final = {"filename": [], "percentage": []}

with open(json_1, "r") as fhandle:
    d_1 = json.load(fhandle)
    
with open(json_2, "r") as fhandle:
    d_2 = json.load(fhandle)
    
with open(json_3, "r") as fhandle:
    d_3 = json.load(fhandle)  
    
with open(json_4, "r") as fhandle:
    d_4 = json.load(fhandle) 
    

    
for key in d_1:
    final_list = d_1[key] + d_2[key] + d_3[key] + d_4[key] 
    d_final["filename"].append(key)
    
    score = sum(final_list) / len(final_list)
    if score >= 100:
        score = 100
    int_score = int(score)
    if score <= 0:
        final_score = 0
    if score - int_score >= 0.5:
        final_score = int_score + 1
    elif (int_score <= score) and (score - int_score < 0.5):
        final_score = int_score
    else:
        if score > 0:
            print("OMG", score, int_score)
    percentage = float(final_score)
    
    d_final["percentage"].append(percentage)


submission_data = pd.DataFrame(data=d_final)
submission_data.to_csv("predictions.csv", header=False, index=False)