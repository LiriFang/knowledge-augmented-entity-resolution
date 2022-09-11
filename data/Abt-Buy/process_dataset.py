import re
import pandas as pd
from collections import defaultdict


def method2_test():
    pattern = re.compile(r'(COL|VAL)')
    s = "COL name VAL transcend jetflash v10 16gb usb flash drive ts16gjfv10 COL description VAL transcend jetflash v10 16gb usb flash drive ts16gjfv10 hi-speed usb 2.0 capless sliding design protects the connector easy plug and play installation aes encryption databackup function keychain hook and neck strap green finish COL price VAL 38.0 	COL name VAL transcend 2gb compactflash card ( 133x ) ts2gcf133 COL description VAL  COL price VAL  	0"
    body = s[:-1]
    tail = s[-1]

    items = pattern.split(body)[1:]
    print(items)
    assert len(items) % 4 == 0
    pairs = []
    for i in range(len(items) // 4):
        pairs.append((items[i*4+1], items[i*4+3]))

    print(pairs[:len(pairs) // 2])
    print(pairs[len(pairs) // 2:])
    print(tail)


def create_sep_ds(col_names, ds):
    dict_prep = dict.fromkeys(col_names, [])
    for list_v in ds:
        for col_name_p, col_value_p in list_v:
            # print({col_name.strip(): col_value})
            # print(col_value)
            col_name = col_name_p.strip()
            col_value = col_value_p.strip()
            dict_prep[col_name].append(col_value)
    return pd.DataFrame(dict_prep)


def method2():
    # input_fn = 'train_exp_convert.txt'
    input_fn = 'test.txt'
    col_names_1 = []
    col_names_2 = []
    ds_1 = []
    ds_2 = []
    ds_3 = []

    for line_p in open(input_fn):
        pattern = re.compile(r'(COL|VAL)')
        # s = "COL name VAL transcend jetflash v10 16gb usb flash drive ts16gjfv10 COL description VAL transcend jetflash v10 16gb usb flash drive ts16gjfv10 hi-speed usb 2.0 capless sliding design protects the connector easy plug and play installation aes encryption databackup function keychain hook and neck strap green finish COL price VAL 38.0 	COL name VAL transcend 2gb compactflash card ( 133x ) ts2gcf133 COL description VAL  COL price VAL  	0"
        line = line_p.rstrip()
        body = line[:-1]
        tail = line[-1]
        print(tail)
        items = pattern.split(body)[1:]
        assert len(items) % 4 == 0
        pairs = []
        for i in range(len(items) // 4):
            pairs.append((items[i * 4 + 1], items[i * 4 + 3]))
        pair_1 = pairs[:len(pairs) // 2]
        pair_2 = pairs[len(pairs) // 2:]
        col_names_1 = [v[0].strip() for v in pair_1]
        col_names_2 = [v[0].strip() for v in pair_2]
        ds_1.append(pair_1)
        ds_2.append(pair_2)
    df_1 = create_sep_ds(col_names_1, ds_1)
    df_2 = create_sep_ds(col_names_2, ds_2)
    df_3 = pd.DataFrame({'flag': ds_3})
    return df_1, df_2, df_3

def str2col(s):
    col_names = []
    val_list = []
    columns = s.split('COL')
    # print(columns)
    for col in columns[1:]:
        vals = col.split('VAL')
        # print(vals)
        col_name = vals[0].strip()
        val = vals[1].strip()


        col_names.append(col_name)
        val_list.append(val)
    assert len(col_names) == len(val_list), "number of col_name is different from the number of vals"
    return col_names, val_list


def method3(input_fn:str):
    col_names_1 = []
    col_names_2 = []
    ds_1 = defaultdict(list)
    ds_2 = defaultdict(list)
    i = 0

    for line_p in open(input_fn):
        line = line_p.strip()
        items = line.split('\t')
        # print(len(items))
        col_names_1, val_list_1 = str2col(items[0])
        col_names_2, val_list_2 = str2col(items[1])

        ds_1[i] = val_list_1
        ds_2[i] = val_list_2
        i+=1
    df_1 = pd.DataFrame.from_dict(ds_1, orient='index', columns=col_names_1)
    df_2 = pd.DataFrame.from_dict(ds_2, orient='index', columns=col_names_2)
    prefix = input_fn.split('.')[0]
    df_1.to_csv(prefix+'-1.csv')
    df_2.to_csv(prefix+'-2.csv')
    return df_1, df_2

def main1():
    pattern_1 = re.compile(r'\s*((COL\s+(.*?)\s+VAL\s+(.*?)\s+)*)([01])$')
    pattern_2 = re.compile(r'COL\s+(.*?)\s+VAL\s+(.*?)\s+')
    s = "COL name VAL transcend jetflash v10 16gb usb flash drive ts16gjfv10 COL description VAL transcend jetflash v10 16gb usb flash drive ts16gjfv10 hi-speed usb 2.0 capless sliding design protects the connector easy plug and play installation aes encryption databackup function keychain hook and neck strap green finish COL price VAL 38.0 	COL name VAL transcend 2gb compactflash card ( 133x ) ts2gcf133 COL description VAL  COL price VAL  	0"
    pattern = re.compile(r'(COL\s+?(.*?)\s+?VAL\s+?(.*?)\s+?)+')
    body = s[:-1]
    tail = s[-1]

    pairs = []
    while match := pattern.fullmatch(body):
        pairs.append((match[2], match[3]))
        body = body[: -len(match[1])]
    pairs = [*reversed(pairs)]

    print(pairs[:len(pairs) // 2])
    print(pairs[len(pairs) // 2:])
    print(tail)


def main():
    # df_1, df_2, df_3 = method2()

    df_1, df_2 = method3('test.txt')
    print('df_1')
    print(df_1.head(), len(df_1))
    df_1.to_csv('test_1.csv')
    print('df_2')
    print(df_2.head(), len(df_2))
    # print('df_3')
    # print(df_3.head(), len(df_3))


if __name__ == '__main__':
    # method2_test()
    # method2()
    main()
