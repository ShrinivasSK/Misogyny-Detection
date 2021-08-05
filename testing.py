from CNN_GRU import CNN_GRU
from LSTM import LSTM
from bert import BERT
from classic_models import Classic
import pandas as pd


def load_dataset(path, index, train_cnt):
    df_train = pd.read_csv(path+'train_'+str(index)+'.csv')
    df_val = pd.read_csv(path+'val_'+str(index)+'.csv')
    df_test = pd.read_csv(path+'test_'+str(index)+'.csv')

    df_train_hate = df_train[df_train['Label'] == 1].sample(train_cnt)
    df_train_non_hate = df_train[df_train['Label'] == 0].sample(train_cnt)

    df_train = pd.concat([df_train_hate, df_train_non_hate])

    df_train = df_train.sample(frac=1).reset_index(drop=True)

    return df_train, df_val, df_test


def choose_model(name, index, train_cnt):
    args = argsAll[name]

    args['name'] = str(index)+'_'+str(train_cnt)

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


def train(model_name, data_path, index, train_cnt,
        res_base_path,all_test_metrics):
    model, args = choose_model(model_name)
    df_train, df_val, df_test = load_dataset(data_path, 
                    index, train_cnt)

    if(model_name=='classic'):
        test_metrics = model.run(args,df_train,df_test)
    else:
        train_metrics, test_metrics = model.run(args, 
                        df_train, df_val, df_test)
        
        # Save train metrics after generating path
        res_path=res_base_path+model_name+'_'+args['name']
        save_metrics(res_path,train_metrics,"train")
    
    test_metrics['name']=args['name']
    
    all_test_metrics.append(test_metrics)
        


sizes = [32, 64, 128, 256, 512]
models = ['lstm', 'cnn_gru', 'bert','classic']
folds = [1, 2, 3, 4, 5]

def run(args):
    all_test_metrics=[]
    
    for fold in folds:
        train(args['model_name'],args['data_path'],fold,
        args['train_cnt'],args['res_base_path'],all_test_metrics)
    
    save_metrics(args['res_base_path']+args['model_name']+
            '_'+str(args['train_cnt'],all_test_metrics,"test"))


argsAll = {
    'lstm': {
        'seed_val': 42,
        'batch_size': 32,
        'max_len': 1024,
        'weights': [1.0, 1.0],
        'epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cpu',
        'embedding_path': "Embeddings/glove.6B.100d.txt",
    },
    'bert': {
        'seed_val': 42,
        'batch_size': 32,
        'bert_model': "bert-base-multilingual-cased",
        'learning_rate': 2e-5,
        'epochs': 4,
        'max_len': 100,
        'device': 'cpu',
        'weights': [1.0, 1.0]
    },
    'cnn_gru': {
        'seed_val': 42,
        'embedding_path': "Embeddings/wiki.multi.en.vec",
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 10,
        'device': 'cpu',
        'model': {
            'train_embed': False,
            'weights': [1.0, 1.0],
        }
    },
    'classic':{
        'embedding': 'bow',
        'top_k': 20000,
        'seed_val':42,
        'weights': [1.0,1.0],
    }
}

run_args={
    'model_name':'bert',
    'data_path':'Data_Processed/Shared_Task_eng/',
    'train_cnt':32,
    'res_base_path': 'Results/Shared_Task_eng/',
}

# run(run_args)