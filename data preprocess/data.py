import pandas as pd
from tools import *
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm

int_feat = ['locdt', 'contp', 'etymd', 'mcc', 'ecfg', 'insfg', 'iterm', 'bnsfg', 'flam1', 'stocn', 'scity', 'stscd', 'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam', 'flg_3dsmk']
cat_feat = ["txkey", "chid", "cano", "contp", "etymd", "mchno", "acqic", "mcc", "ecfg", "insfg", "bnsfg", "stocn", "scity", "stscd", "ovrlt", "flbmk", "hcefg", "csmcu", "flg_3dsmk"]
class tbrain_data:
    def __init__(self, split = 'train', preprocess = True) -> None:
        self.split = split
        
        if split == 'trainval':
            self.data = pd.concat([pd.read_csv(f'./data/train.csv'), pd.read_csv(f'./data/val.csv')])
        else:
            self.data = pd.read_csv(f'./data/{split}.csv')
        # l = self.data.columns.values.tolist()
        # kk = self.data[:10]
        # d = [self.data[each][1] for each in l]
        # ddd = self.data.label
        # print()
        if preprocess: self.preprocess()

    def preprocess(self):
        self.data['loctm'].apply(func=time2seconds)
        for each in int_feat:
            self.data[each] = self.data[each].fillna(value=-1).astype(dtype=int)

        if self.split == 'train' or self.split == 'trainval':
            self.data["label"] = self.data["label"].fillna(value=0).astype(dtype=int)
            self.data["label"] = self.data["label"].replace(-1, 0)
    def split_xy(self):
        number_k = ['locdt','loctm','chid',
                    'cano','contp','etymd',
                    'mchno','acqic','mcc',
                    'conam','ecfg','insfg',
                    'iterm','bnsfg','flam1',
                    'stocn','scity','stscd',
                    'ovrlt','flbmk','hcefg',
                    'csmcu','csmam','flg_3dsmk']
        # i = self.data['txkey']
        x = pd.DataFrame(self.data, columns=number_k)
        x = x[:-1]
        y = None
        if self.split == 'train':
            # cols = self.data.columns.values.tolist()
            y = self.data['label']
            # y = y[:-1].astype(str)
        elif self.split == 'test':
            pass
        return x, y
    
    def pack_to_catboost(self):
        
        structured_data = self.data
        if self.split in ["train", "trainval"]:
            structured_data = Pool(data=structured_data.drop(labels="label", axis=1),
                                label=structured_data.label,
                                cat_features=cat_feat)
        else:
            structured_data = Pool(data=structured_data,
                                cat_features=cat_feat)

        return structured_data


    def to_csv(self, dir):
        self.data.to_csv(dir)

def sort_by_users(train, val, test):
    train = train.data
    val = val.data
    test = test.data
    # split_record = {}
    # for 

    train = train.sort_values(by=['chid', 'cano', 'locdt', 'loctm'])
    groups = [g[1] for g in train.groupby('cano')]
    for i in tqdm(range(len(groups))):
        groups[i].insert(groups[i].shape[1]-1, 'last_label', 0)
    for i in tqdm(range(len(groups))):
        for j in range(groups[i].shape[0]):
            if j == 0: groups[i].loc[i,'last_label'] = 0
            else:
                # print(int(g.label[i-1])) 
                groups[i].loc[j,'last_label'] = groups[i].loc[j-1,'label']
    train = pd.concat(groups)

if __name__ == "__main__":
    train = tbrain_data('train')
    val = tbrain_data('validate')
    test = tbrain_data('test')
    train, val, test = sort_by_users(train, val, test)
    train.to_csv('./data/train_sorted.csv')
    val.to_csv('./data/val_sorted.csv')
    test.to_csv('./data/test_sorted.csv')
