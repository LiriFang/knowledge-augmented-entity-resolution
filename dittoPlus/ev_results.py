import json

with open('result.json', 'r')as f:
    data = json.load(f)

with open('../prompt=space/result.json', 'r')as f1:
    data1 = json.load(f1)


# same knowledge injection method, different prompting method
#  data: prompt with slash; data1: prompt with space 
rows = data['rows']
rows1 = data1['rows']
assert len(rows) == len(rows1)

for i, row in enumerate(rows):
    row1 = rows1[i]
    gd = row['ground_truth']
    pred_res = row['pred_result']
    pred_res1 = row1['pred_result']
    # x, x' --> y, y'

