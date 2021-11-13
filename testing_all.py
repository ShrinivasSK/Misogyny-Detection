from data_cleaning import Data_Preprocessing
from CNN_GRU import CNN_GRU
from LSTM import LSTM
from bert import BERT
from classic_models import Classic
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm

from nltk.corpus import stopwords

# from arabert.preprocess import ArabertPreprocessor

# fix the random
def fix_random(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# load dataset 
# keep count of hate and non_hate data
# equal to train_cnt in train set 
def load_dataset(args,index):
    # initialise constants 
    path = args['data_path']
    train_cnt = args['train_cnt']
    model_name = args['model_name']
    # read dataframes
    df_train = pd.read_csv(path+'train_'+str(index)+'.csv')
    df_val = pd.read_csv(path+'val_'+str(index)+'.csv')
    df_test = pd.read_csv(path+'test_'+str(index)+'.csv')

    # clean data
    df_train=clean_data(df_train,model_name)
    df_val=clean_data(df_val,model_name)
    df_test=clean_data(df_test,model_name)

    return df_train, df_val, df_test


def clean_data(df,model_name):
    X = df['Text']
#     prep = ArabertPreprocessor('bert-base-arabertv02')
    processer = Data_Preprocessing()
    X_new=[]
    if(model_name!='bert'):
        stop_words = set(stopwords.words('spanish'))
        for text in tqdm(X):
            text= processer.removeEmojis(text)
            text = processer.removeUrls(text)
            text= processer.removeSpecialChar(text)
            text = processer.removeStopWords(text,stop_words)
            text= processer.lemmatise(text)
            X_new.append(text)
    else:
        for text in tqdm(X):
            text= processer.removeEmojis(text)
            text = processer.removeUrls(text)
            text=processer.removeSpecialChar(text)
            X_new.append(text)
#     for text in tqdm(X):
#         text = prep.preprocess(text)
#         X_new.append(text)
    df['Text']=X_new
    return df 

# Choose model
def choose_model(name, index, train_cnt,run_args):
    args = argsAll[name]

    args['name'] = str(index)+'_'+str(train_cnt)
    args['model_save_path']=run_args['model_save_path']

    if(name == 'lstm'):
        model = LSTM(args)
    elif(name == 'cnn_gru'):
        model = CNN_GRU(args)
    elif(name == 'bert'):
        model = BERT(args)
    elif(name=='classic'):
        model=Classic(args)

    return model, args

def save_metrics(path,metrics,which):
    df = pd.DataFrame(metrics)
    df.to_csv(path+"_"+which+".csv")


def train(args, index,all_test_metrics):
    model_name = args['model_name']
    print("\tInitialising Model....")
    model, model_args = choose_model(model_name,index,args['train_cnt'],args)
    print("\tLoading Dataset....")
    df_train, df_val, df_test = load_dataset(args,index)
    print("\tTraining Starts....")
    if(model_name=='classic'):
        test_metrics = model.run(model_args,df_train,df_test)
    else:
        train_metrics, test_metrics = model.run(model_args, 
                        df_train, df_val, df_test)
        
        # Save train metrics after generating path
#         res_path=args['res_base_path']+model_name+'_'+model_args['name']
#         save_metrics(res_path,train_metrics,"train")
    
    test_metrics['name']=model_args['name']
    
    all_test_metrics.append(test_metrics)

sizes = [32, 64, 128, 256, 512]
models = ['lstm', 'cnn_gru', 'bert','classic']
folds = [1, 2, 3, 4, 5]
seeds = [43,44,45]

def run(args):
    all_test_metrics=[]
    
#     for fold in folds:
    fold=1
    print("Fold: ",fold)
    fix_random()
    train(args,fold,all_test_metrics)
#     print("Saving Test Metrics....")
#     save_metrics(args['res_base_path']+args['model_name']+
#             '_'+str(args['train_cnt']),all_test_metrics,"test")

argsAll = {
    'lstm': {
        'seed_val': 42,
        'batch_size': 8,
        'max_len': 128,
        'weights': [1.0, 1.0],
        'epochs': 20,
        'learning_rate': 1e-4,
        'device': 'cuda',
        'embedding_path': "Embeddings/cc.ar.300.vec",
        'save_model': False,
    },
    'bert': {
        'seed_val': 42,
        'batch_size': 8,
#         'bert_model': "aubmindlab/bert-base-arabertv02",
        'bert_model': "bert-base-multilingual-cased",
#         'bert_model': "bert-base-uncased",
        'learning_rate': 2e-5,
        'epochs': 10,
        'max_len': 128,
        'device': 'cuda',
        'weights': [1.0, 1.0],
        'save_model': True,
    },
    'cnn_gru': {
        'seed_val': 42,
        'embedding_path': "Embeddings/cc.ar.300.vec",
        'batch_size': 8,
        'learning_rate': 1e-4,
        'epochs': 20,
        'device': 'cuda',
        'max_len': 128,
        'model': {
            'train_embed': False,
            'weights': [1.0, 1.0],
        },
        'save_model': False,
    },
    'classic':{
        'embedding': 'bow',
        'top_k': 20000,
        'seed_val':42,
        'weights': [1.0,1.0],
        'save_model': False,
    }
}

run_args={
    'model_name':'bert',
    'data_path':'Data_Processed/AMI-2020/',
    'train_cnt':256,
    'res_base_path': 'Results/AMI-2020/all/',
    'model_save_path': 'Saved_Models/AMI-2020/',
}

# for model in models[:-1]:
model='bert'
print(model)
run_args['model_name']=model
print('Data: ',run_args['data_path'].split('/')[-2])
run_args['train_cnt']='all'
run(run_args)