import json

# no knowledge injection 
with open('./output/Structured/iTunes-Amazon/None/prompt=space/result.json', 'r')as f:
    data = json.load(f)

# with knowledge (sherlock)...
with open('./output/Structured/iTunes-Amazon/sherlock/prompt=space/result.json', 'r')as f1:
    data_aug = json.load(f1)


# same knowledge injection method, different prompting method
#  data: prompt with slash; data1: prompt with space 
rows = data['rows']
rows_aug = data_aug['rows']
assert len(rows) == len(rows_aug)

false_pred = 0
false_pred_aug = 0
for i, row in enumerate(rows):
    row_aug = rows_aug[i]
    # print(f"original data inputs left: {row['left']}")
    # print(f"original data inputs right: {row['right']}")
    gd = int(row['ground_truth'])
    pred_res = int(row['pred_result'])
    pred_res_aug = int(row_aug['pred_result'])
    # x, x' --> y, y'
    if pred_res_aug==gd and pred_res!=gd:
        print(f'Improved row index: {i}')
        print(f"Without knowledge injection, predicted result: {row['pred_result']}")
        print(f"With knowledge injection, predicted result: {row_aug['pred_result']}")
    if pred_res_aug!=gd:
        # print(f'wrong predict row with KA: {i}')
        false_pred_aug += 1 
        # print(f'False predicted row: {i}')
        # print(f'Matching ground truth: {gd}')
    if pred_res!=gd:
        print(f'wrong predict row without KA: {i}')
        false_pred += 1

print(f'Total wrong predicted rows without KA: {false_pred}')
print(f'Total wrong predicted rows with KA: {false_pred_aug}')


