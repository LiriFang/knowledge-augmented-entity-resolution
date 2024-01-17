import json
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def read_res(file_path):
    with open(file_path, 'r')as f:
        data = json.load(f)
    return data 


def cosine_similarity(list1, list2):
    dot_product = np.dot(list1, list2)
    norm_list1 = np.linalg.norm(list1)
    norm_list2 = np.linalg.norm(list2)
    
    similarity = dot_product / (norm_list1 * norm_list2)
    
    return similarity


def doc_distance(res_sherlock, res_doduo, details_df, idx_same=4, idx_diff=[1,2,5,6]):
    # the distance between vectors should represent the similarity between the data entries
    # predicted_wt_sherlock: T, predicted_wt_doduo: T
    # predicted_wt_sherlock: F, predicted_wt_doduo: T
    # cos_(v1, v2)
    # 'row_index', 'entry_sherlock', 'entry_doduo', 'predict_sherlock','predict_doduo', 'distance', 'ground_truth'
    delta_res = []
    rows_sherlock = res_sherlock['rows']
    rows_doduo = res_doduo['rows']
    predict_diff_rows = []
    predict_same_rows = details_df.iloc[idx_same]['Rows Indices']
    for idx in idx_diff:
        predict_diff_rows.extend(details_df.iloc[idx]['Rows Indices'])
    # values_sherlock = []
    # values_doduo = []
    distances = {}
    exp_rows = predict_same_rows + predict_diff_rows
    for row_idx in exp_rows:
        row_sherlock = rows_sherlock[row_idx]
        entry_sherlock = row_sherlock['left'] + ' ' + row_sherlock['right']
        predict_sherlock = row_sherlock['pred_result']
        vec_sherlock = row_sherlock['vectors']
        gd_sherlock = row_sherlock['ground_truth']

        row_doduo = rows_doduo[row_idx]
        entry_doduo = row_doduo['left'] + ' ' + row_doduo['right']
        predict_doduo = row_doduo['pred_result']
        vec_doduo = row_doduo['vectors']
        gd_doduo = row_doduo['ground_truth']

        assert gd_sherlock==gd_doduo

        vec_sim = cosine_similarity(vec_sherlock, vec_doduo)
        print(f'at row {row_idx}: predicted result by sherlock is {predict_sherlock}, by doduo is {predict_doduo}, cos sim: {vec_sim}, ground truth: {gd_sherlock}')
        delta_row = [row_idx, entry_sherlock, entry_doduo, predict_sherlock, predict_doduo, vec_sim, gd_sherlock]
        delta_res.append(delta_row)
    return delta_res


def write_ev_table(data_no_ka, data_sherlock, data_doduo, data_el=None):
    '''
    #@params data_no_ka: data without knowledge augmentation
    #@params data_sherlock: data with column semantic types predicted by sherlock
    #@params data_doduo: data with column semantic types predicted by doduo
    #@params data_el: data with entity linking 
    '''
    rows_index_details = []
    values = []
    rows = data_no_ka['rows']
    rows_sh = data_sherlock['rows']
    rows_doduo = data_doduo['rows']
    assert len(rows) == len(rows_sh) == len(rows_doduo)

    true_all = 0
    true_all_rows = []
    false_doduo = 0
    false_doduo_rows = []
    false_sh = 0
    false_sh_rows = []
    false_both_ka = 0
    false_both_ka_rows = []
    false_no_ka = 0
    false_no_ka_rows = []
    true_sh = 0
    true_sh_rows = []
    true_doduo = 0
    true_doduo_rows = []
    false_all = 0
    false_all_rows = []

    for i, row in enumerate(rows):
        row_sh = rows_sh[i]
        row_doduo = rows_doduo[i]
        gd = int(row['ground_truth'])
        pred_res = int(row['pred_result'])
        pred_res_sh = int(row_sh['pred_result'])
        pred_res_doduo = int(row_doduo['pred_result'])
        # x, x' --> y, y'
        if pred_res_sh==gd and pred_res_doduo==gd and pred_res==gd:
            true_all += 1
            true_all_rows.append(i)
        
        if pred_res==gd and pred_res_sh==gd and pred_res_doduo!=gd:
            false_doduo += 1
            false_doduo_rows.append(i)

        if pred_res==gd and pred_res_sh!=gd and pred_res_doduo==gd:
            false_sh += 1
            false_sh_rows.append(i)
        
        if pred_res==gd and pred_res_sh!=gd and pred_res_doduo!=gd:
            false_both_ka += 1
            false_both_ka_rows.append(i)

        if pred_res!=gd and pred_res_sh==gd and pred_res_doduo==gd:
            false_no_ka += 1
            false_no_ka_rows.append(i)

        if pred_res!=gd and pred_res_sh==gd and pred_res_doduo!=gd:
            true_sh += 1
            true_sh_rows.append(i)

        if pred_res!=gd and pred_res_sh!=gd and pred_res_doduo==gd:
            true_doduo += 1
            true_doduo_rows.append(i)

        if pred_res!=gd and pred_res_sh!=gd and pred_res_doduo!=gd: 
            false_all += 1
            false_all_rows.append(i) 
    true_all_ratio = true_all/len(rows)
    values.append(['T', 'T', 'T', true_all, true_all_ratio])
    rows_index_details.append(['Predicted results are all correct: with or without KA', true_all_rows])
    false_doduo_ratio = false_doduo/len(rows)
    values.append(['T', 'T', 'F', false_doduo, false_doduo_ratio])
    rows_index_details.append(['Knowledge augmented by Doduo worse the predicted results', false_doduo_rows])
    false_sh_ratio = false_sh/len(rows)
    values.append(['T', 'F', 'T', false_sh, false_sh_ratio])
    rows_index_details.append(['Knowledge augmented by Sherlock worse the predicted results', false_sh_rows])
    false_both_ka_ratio = false_both_ka/len(rows)
    values.append(['T', 'F', 'F', false_both_ka, false_both_ka_ratio])
    rows_index_details.append(['Knowledge augmented by both methods worse the predicted results', false_both_ka_rows])
    false_no_ka_ratio = false_no_ka/len(rows)
    values.append(['F', 'T', 'T', false_no_ka, false_no_ka_ratio])
    rows_index_details.append(['Knowledge augmented by both methods improve the predicted results', false_no_ka_rows])
    true_sh_ratio = true_sh/len(rows)
    values.append(['F', 'T', 'F', true_sh, true_sh_ratio])
    rows_index_details.append(['Knowledge augmented by Sherlock improve the predicted results', true_sh_rows])
    true_doduo_ratio = true_doduo/len(rows)
    values.append(['F', 'F', 'T', true_doduo, true_doduo_ratio])
    rows_index_details.append(['Knowledge augmented by Duoduo improve the predicted results', true_doduo_rows])
    false_all_ratio = false_all/len(rows)
    values.append(['F', 'F', 'F', false_all, false_all_ratio])
    rows_index_details.append(['Predicted results are all False: with or without KA', false_all_rows])
    # print(values)
    return values, rows_index_details


def main():
    data_no_ka = read_res('./output/Structured/DBLP-ACM/None/prompt=space/result.json')
    data_doduo = read_res('./output/Structured/DBLP-ACM/doduo/prompt=space/result.json')
    data_sherlock = read_res('./output/Structured/DBLP-ACM/sherlock/prompt=space/result.json')

    analysis_fp = 'q_experiment.csv'
    details_fp = 'q_rows.csv'
    if os.path.exists(analysis_fp) and os.path.exists(details_fp):
        print(f'The file path {analysis_fp} and {details_fp} exist.')
        ev_df = pd.read_csv(analysis_fp)
        # dtype_mapping = {'Description': 'str', 'Rows Indices':'list'}
        details_df = pd.read_csv(details_fp)
        details_df['Rows Indices'] = details_df['Rows Indices'].apply(json.loads)
    else:
        column_names = ['Predicted result[Without KA]', 'Predicted Result[With Sherlock]', 'Predicted Result[With Doduo]', 'Rows Count', 'Error Ratio']
        col_names_bp = ['Description', 'Rows Indices']
        ev_df = pd.DataFrame(columns=column_names)
        details_df = pd.DataFrame(columns=col_names_bp)
        rows_list, details_list = write_ev_table(data_no_ka, data_sherlock, data_doduo)
        for rows in rows_list:  
            ev_df = ev_df.append(pd.Series(rows, index=column_names), ignore_index=True)
        
        for details in details_list:
            details_df = details_df.append(pd.Series(details, index=col_names_bp), ignore_index=True)
        ev_df.to_csv(analysis_fp, index=False)
        details_df.to_csv(details_fp, index=False)
        # print(f'The file path {analysis_fp} does not exist.')
    
    # x->x'; y->y'
    # delta_df document the distances between data entry vectors 
    rows_delta = doc_distance(data_sherlock, data_doduo, details_df)
    delta_cols = ['row_index', 'entry_sherlock', 'entry_doduo', 'predict_sherlock','predict_doduo', 'similarity', 'ground_truth']
    delta_df = pd.DataFrame(columns=delta_cols)
    for row_delta in rows_delta:
        delta_df = delta_df.append(pd.Series(row_delta, index=delta_cols), ignore_index=True)
    delta_fp = 'q3_delta.csv'
    delta_df.to_csv(delta_fp, index=False)   




if __name__ == '__main__':
    main()