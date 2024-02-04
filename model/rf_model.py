import os, sys 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Dict, Tuple, Union
import numpy as np 
sys.path.append("../")
from utils import analysis_utils 
from eval import evaluate_reveal_predc, evaluate_surv_days, evaluate_rgr, evaluate_clf, format_output, feature_analysis

REVERSE_EVAL = True

def merget_feature_gt(feature_df, gt_df, feature_cols):
  # due to feature computed per chg (not the final status)
  rows =[]
  for _, gt_row in gt_df.iterrows():
    bid, mutK = gt_row['bid'], gt_row['mutK']
    feature_vs = feature_df.loc[(feature_df.bid == bid) & (feature_df.mutK == mutK)][feature_cols].values[0]
    rows.append(gt_row.tolist() + feature_vs.tolist())
  feature_and_gt_df = pd.DataFrame(rows, columns = gt_df.columns.tolist() + feature_cols)
  return feature_and_gt_df

def get_feature_and_gt(feature_dir:str, gtdir:str, project:str, 
  key_cols:List[str], feature_cols:List[str], gt_col:str, thr:float, 
  which:str = 'reveal') -> pd.DataFrame:
  from data import get_data
  feature_df = get_data(feature_dir, project)[key_cols + feature_cols + ['status']]
  if which == 'survive': # for survival prediction -> here, ranking 
    from data import get_gts 
    gt_df = get_gts(gtdir, project, gt_col)
  else: # revealed mutant within xxx days (the same for surv -> "confirmed" to be survived at least xxs days)
    from data import get_gts_for_reveal
    gt_df = get_gts_for_reveal(gtdir, project, gt_col, thr)
  #
  if project == 'Math': 
    gt_df = gt_df.loc[gt_df.bid != 59]
    feature_df = feature_df.loc[feature_df.bid != 59]
  #
  feature_and_gt_df = merget_feature_gt(feature_df, gt_df, feature_cols)
  #feature_and_gt_df = feature_df.merge(
  #  gt_df, on = ['bid', 'mutK'], how = 'inner', suffixes = ("_drop", ""))
  return feature_and_gt_df 

def get_feature_and_gt_all(feature_dir:str, gtdir:str, projects:str, 
  key_cols:List[str], feature_cols:List[str], gt_col:str, thr:float, 
  which:str = 'reveal') -> pd.DataFrame:
  # get features 
  datas = []
  for project in projects:
    data = get_feature_and_gt(feature_dir, gtdir, project, key_cols, feature_cols, gt_col, thr, which = which)
    data['project'] = [project] * len(data)
    datas.append(data)
  merged_data = pd.concat(datas, ignore_index=True)
  return merged_data

def train_and_predict_survived_days(X, y, random_state:int = 0) -> Tuple[List[int], List[float], Dict]:
  # get features and gts 
  rgrs = {}
  kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
  test_idx_l = []
  predictions = []
  for i, (train_idx, test_idx) in enumerate(kf.split(X = X, y = y)):
    test_idx_l.extend(test_idx.tolist())
    regr = RandomForestRegressor(max_depth=None, random_state=random_state)
    train_X, train_y = X[train_idx], y[train_idx]
    test_X = X[test_idx]
    regr.fit(train_X, train_y)
    # predict  
    predcs = regr.predict(test_X) # evaluate_rgr
    predictions.extend(predcs.tolist())
    # store
    rgrs[i] = {'test_idx':test_idx, 'model':regr}
  return test_idx_l ,predictions, rgrs

def train_and_predict_status(X, y, random_state:int = 0, prob:bool = True) -> Tuple[List[int], List[float], Dict]:
  # get features and gts 
  clfs = {}
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
  test_idx_l, predictions = [],[]
  for i, (train_idx, test_idx) in enumerate(skf.split(X = X, y = y)):
    test_idx_l.extend(test_idx.tolist())
    clf = RandomForestClassifier(max_depth=None, random_state=random_state)
    train_X, train_y = X[train_idx], y[train_idx]
    test_X = X[test_idx]
    clf.fit(train_X, train_y)
    # predict  
    if prob:
      predcs = clf.predict_proba(test_X) # or predict # evaluate_clf
    else:
      predcs = clf.predict(test_X)
    #
    train_avg_acc = clf.score(train_X, train_y)
    test_avg_acc = clf.score(test_X, y[test_idx])
    print ("train", train_avg_acc, test_avg_acc)
    #
    #predcs = clf.predict(test_X) # or predict
    predictions.extend(predcs.tolist())
    # store
    clfs[i] = {'test_idx':test_idx, 'model':clf}
  return test_idx_l, predictions, clfs

def main_surv_pred(feature_dir:str, gtdir:str, projects:str, 
  key_cols:List[str], feature_cols:List[str], random_state:int = 0): 
  # all: get data
  data = get_feature_and_gt_all(feature_dir, gtdir, projects, key_cols, feature_cols, 'debt_time', None, which = 'survive') 
  test_idx_l, predictions, rgrs = train_and_predict_survived_days(
    data[feature_cols].values, data[gt_col].values, random_statie = random_state)
  # evaluate 
  data_in_test_order = data.iloc[test_idx_l]
  data_in_test_order.loc[:,'pred'] = predictions 
  maps = evaluate_surv_days(data_in_test_order, 'debt_time', REVERSE_EVAL = REVERSE_EVAL)
  return rgrs, maps 

def random_sel_surv_pred(feature_dir:str, gtdir:str, projects:str, key_cols:List[str], random_state:int = 0):
  # all: get data
  data = get_feature_and_gt_all(feature_dir, gtdir, projects, key_cols, [], 'debt_time', None, which = 'survive') 
  # evaluate 
  np.random.seed(random_state)
  predcs = np.random.rand(len(data))
  predcs /= predcs.sum()
  data['pred'] = predcs
  maps = evaluate_surv_days(data, 'debt_time', REVERSE_EVAL = REVERSE_EVAL)
  return None, maps 

# 
def main_reveal_pred(feature_dir:str, gtdir:str, projects:List[str], 
  key_cols:List[str], feature_cols:List[str], gt_col:str, 
  thr:float, prob:bool = True, random_state:int = 0, idx_to_reveal_label:int = 0): 
  # all: get data
  data = get_feature_and_gt_all(feature_dir, gtdir, projects, key_cols, feature_cols, gt_col, thr, which = 'reveal') 
  #
  test_idx_l, predictions, clfs = train_and_predict_status(
    data[feature_cols].values, data['label'].values, random_state = random_state, prob = prob)
  # evaluate 
  data_in_test_order = data.iloc[test_idx_l]
  data_in_test_order['pred'] = predictions 
  bal_accs, maps = evaluate_reveal_predc(data_in_test_order, prob = prob, idx_to_reveal_label = idx_to_reveal_label)
  return clfs, (bal_accs, maps), data_in_test_order

def random_reveal_pred(feature_dir:str, gtdir:str, projects:List[str], 
  key_cols:List[str], gt_col:str, 
  thr:float, prob:bool = True, random_state:int = 0, idx_to_reveal_label:int = 0, 
  as_ZeroR:bool = False, per_project:bool = False): 

  def pred_by_zeroRule(data:pd.DataFrame, per_project:bool = False, prob:bool = False) -> Union[Dict, np.ndarray]:
    """
    reveal:0, surv:1, dead:2 
    """
    label_kv = {'reveal':0, 'surv': 1, 'dead':2}
    if per_project:
      predcs = {}
      for project, df_proj in data.groupby('project'):
        label_freq = {0:0, 1:0, 2:0}
        cnt_df = df_proj.groupby('status')['status'].count()
        for label_k, label_v in cnt_df.items():
          label_freq[label_kv[label_k]] = label_v/len(df_proj)
        # to further ensure the sum to 1
        label_freq[2] = 1. - label_freq[0] - label_freq[1]
        _predcs = np.random.choice(3, size = len(df_proj), p = [label_freq[0], label_freq[1], label_freq[2]])
        if prob:
          _prob_predcs = np.zeros((len(df_proj), 3))
          for i,_pred in enumerate(_predcs):
            _prob_predcs[i,_pred] = 1. 
          _predcs = list(_prob_predcs)
        #if prob:
        #  #_predcs /= len(df_proj)
        #  pred_probs = []
        #  ...
        predcs[project] = _predcs
        #predcs = np.append(predcs, proj_predcs)
      return predcs  
    else: # five fold -> we will simply take all 
      label_freq = {0:0, 1:0, 2:0}
      cnt_df = data.groupby('status')['status'].count()
      for label_k, label_v in cnt_df.items():
        label_freq[label_kv[label_k]] = label_v 
      # to further ensure the sum to 1
      label_freq[2] = 1. - label_freq[0] - label_freq[1]
      predcs = np.random.choice(3, size = len(data), p = [label_freq[0], label_freq[1], label_freq[2]])
      if prob:
        prob_predcs = np.zeros((len(data), 3))
        for i, pred in enumerate(predcs):
          prob_predcs[i, pred] = 1. 
        predcs = list(prob_predcs)
      return predcs  

  # all: get data
  data = get_feature_and_gt_all(feature_dir, gtdir, projects, key_cols, [], gt_col, thr, which = 'reveal') 
  # evaluate 
  np.random.seed(random_state)
  if not as_ZeroR:
    if prob:
      predcs = np.random.dirichlet([1,1,1], size = len(data))
    else:
      predcs = np.random.choice(3, len(data))
  else:
    predcs = pred_by_zeroRule(data, per_project = per_project, prob = prob)
  
  if not per_project:
    data['pred'] = predcs
  else:
    #for p,v in predcs.items():
    #  print (p, len(v))
    data_w_predcs = []
    for project, df_proj in data.groupby('project'):
      #print (df_proj.shape)
      #print (predcs[project].shape)
      df_proj['pred'] = predcs[project]
      data_w_predcs.append(df_proj)
    data = pd.concat(data_w_predcs)
  print (data)
  bal_accs, maps = evaluate_reveal_predc(data, prob = prob, idx_to_reveal_label = idx_to_reveal_label)
  return None, (bal_accs, maps), data

def get_destdir(dest:str, featureK:str, modelK:str) -> str:
  dest = os.path.join(dest, modelK, featureK)
  os.makedirs(dest, exist_ok=True)
  return dest

def save_rf_model(models:Dict, dest:str) -> str:
  import pickle 
  model_file = os.path.join(dest, "rf_model.sav")
  with open(model_file, 'wb') as f:
    pickle.dump(models, f)
  return model_file

def save_output(output:pd.DataFrame, k:str, dest:str) -> str:
  outputfile = os.path.join(dest, f"{k}_pred.csv")
  output.to_csv(outputfile, index=False)
  return outputfile


if __name__ == "__main__":
  import argparse 
  parser = argparse.ArgumentParser()
  parser.add_argument("-p", "--project", type = str)
  parser.add_argument("-d", "--dest", type = str)
  parser.add_argument("-gt", "--gtdir", type = str, default = "../output/evaluation/processed")#combined_v3")
  parser.add_argument("-feature", "--feature_dir", type = str, default = "../output/evaluation/features")#_v3")
  parser.add_argument("-thr", "--threshold", type = int, default = 365)
  parser.add_argument("-rd", "--compute_random", action ="store_true")
  parser.add_argument("-s", "--seed", type = int, default = 0)
  parser.add_argument("-c", "--feature_config", type = str, default = "all")
  args = parser.parse_args() 

  project = args.project 
  #feature_dir = args.feature_dir 
  #gtdir = args.gtdir
  feature_dir = '/Users/jeongju.sohn/workdir/mutBugInducing/output/evaluation/features_v3'
  gtdir = '/Users/jeongju.sohn/workdir/mutBugInducing/output/evaluation/combined_v3'
  dest = args.dest 
  os.makedirs(dest, exist_ok=True)

  if args.project == 'all':
    projects = ['Lang', 'Math', 'Time', 'Closure', 'Cli', 'Compress', 'Codec', 'Collections', 'Csv', 
    'JacksonCore', 'JacksonXml', 'JxPath', 'Jsoup']
  else:
    projects = [project]

  feature_cols_opt = {
    'all':[
      'mutOp','l_churn',
      'l_min_age', 'l_max_age', 'l_n_authors', 'e_churn', 'e_min_age',
      'e_max_age', 'e_n_authors'], 
    'wo_mutop':[
      'l_churn',
      'l_min_age', 'l_max_age', 'l_n_authors', 'e_churn', 'e_min_age',
      'e_max_age', 'e_n_authors'
    ], 
    'wo_lchgs':[
      'mutOp', 
      'e_churn', 'e_min_age',
      'e_max_age', 'e_n_authors'
    ], 
    'wo_echgs':[
      'mutOp',
      'l_churn',
      'l_min_age', 'l_max_age', 'l_n_authors', 
    ], 
    'only_mutop':['mutOp']
  }
  gt_col = 'debt_time' 
  threshold = args.threshold #365
  random_state =args.seed
  
  featureK = args.feature_config # "all" #"wo_mutop" #"all"
  modelK = f"{gt_col}_thr{threshold}"
  prob = True

  if args.compute_random:
    print ("=============================================")
    print ("==================Random=====================")
    print ("=============================================")

    _, _, rd_output = random_reveal_pred(feature_dir, gtdir, projects, 
      ['bid', 'mutK'], gt_col, threshold, prob = prob, random_state = random_state, idx_to_reveal_label = 0, 
      # new
      as_ZeroR=True, 
      per_project = True
    )
    formatted_output = format_output(rd_output, gt_col)
    dest = get_destdir(dest, featureK, 'rd')
    dest = os.path.join(dest, str(args.seed))
    os.makedirs(dest, exist_ok=True)
    outputfile = save_output(formatted_output, 'rd', dest)
    print (f"Save to {outputfile}")
  else:
    clfs, (bal_accs, maps), output = main_reveal_pred(
      feature_dir, gtdir, projects, ['bid', 'mutK'], feature_cols_opt[featureK],
      gt_col, threshold, prob = prob, random_state = random_state, idx_to_reveal_label = 0)
    #print('original', output.label.unique())
    # feature analaysis
    top_n = 5
    for fold_idx, test_idx_and_model in clfs.items():
      print ("=============================================")
      print (f"In fold {fold_idx}")
      test_idx = test_idx_and_model['test_idx']
      mdl = test_idx_and_model['model']
      feature_analysis(mdl, feature_cols_opt[featureK], output, top_n = top_n)
      #print (output.label.unique())
      print ("=============================================")

    dest = get_destdir(dest, featureK, modelK)
    dest = os.path.join(dest, str(args.seed))
    os.makedirs(dest, exist_ok=True)
    model_file = save_rf_model(clfs, dest)
    print (f"Save to {model_file}")
    formatted_output = format_output(output, gt_col)
    outputfile = save_output(formatted_output, 'rf', dest)
    print (f"Save to {outputfile}")
  # save
  
  ## also, used ...
  #if args.compute_random:
    #print ("=============================================")
    #print ("==================Random=====================")
    #print ("=============================================")
    #_, _, rd_output = random_reveal_pred(feature_dir, gtdir, projects, 
      #['bid', 'mutK'], gt_col, threshold, prob = prob, random_state = random_state, idx_to_reveal_label = 0)
    #formatted_output = format_output(rd_output, gt_col)
    #outputfile = save_output(formatted_output, 'rd', dest)
    #print (f"Save to {outputfile}")
  
