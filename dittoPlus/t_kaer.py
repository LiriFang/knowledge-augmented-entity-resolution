import json

# every dataset generate a logging file:
# '{dataset}-{method}.json'
# 'structured/iTunes-Amazon-cst-0-0.json'
logging_info = {
    'dataset-path': 'Structured/iTunes-Amazon/test.txt',
    'rows':[
     {
    'cell_value': "COL Song_Name VAL Elevator ( feat . Timbaland ) COL Artist_Name VAL Flo Rida COL Album_Name VAL Mail On Sunday ( Deluxe Version ) COL Genre VAL Hip-Hop/Rap , Music , Dirty South COL Price VAL $ 1.99 COL CopyRight VAL 2008 Atlantic Recording Corporation for the United States and WEA International Inc. for the world outside of the United States COL Time VAL 3:55 COL Released VAL 17-Mar-08 	COL Song_Name VAL Money Right ( feat . Rick Ross & Brisco ) [ Explicit ] COL Artist_Name VAL Flo Rida COL Album_Name VAL Mail On Sunday [ Explicit ] COL Genre VAL Rap & Hip-Hop COL Price VAL $ 1.29 COL CopyRight VAL 2013 Warner Bros. . Records Inc. COL Time VAL 3:17 COL Released VAL March 17 , 2008 	0",
    'CST': True,
    'CST_values': {
        'Song_Name': '',
        'Artist_Name': '',
        'Album_Name': '',
        'Genre': '',
        'Price': '',
        'Copyright': ''
    },
    'prompting_method': ' ',
    'kbert': False,
    'EL': False,
    'ground truth': 0,
    'predicted result': 0,
    'vector(before)': [],
    'vector(after)': []
    },
    {

        
    }
    ]
    
    }

def save(data, file):
    with open(file, 'wt')as f:
        json.dump(data, f)
    

def main():
    external_users: list[dict] = [
        users
        for user in users.values()
        if user.get('external')
    ]
    for e_user in external_users:
        e_user['enabled'] = False
    save(users, 'users.json')


if __name__ == '__main__':
    main()