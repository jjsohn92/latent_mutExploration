import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import rankdata 
from scipy.stats import spearmanr 
import numpy as np  

def load_models(model_path:str):
  import pickle 
  with open(model_path, 'rb') as f:
    mdl = pickle.load(f)
  models = {}
  test_indices = {}
  for foldIdx, mdl_and_indices in mdl.items():
    models[foldIdx] = mdl_and_indices['model']
    test_indices[foldIdx] = mdl_and_indices['test_idx']
  return models, test_indices

def evaluate_rgr(regr, X) -> np.ndarray:
  predcs = regr.predict(X)
  return predcs  

def evaluate_clf(clf, X, prob:bool = True) -> np.ndarray:
  if prob:
    predcs = clf.predict_proba(X) # or predict 
  else:
    predcs = clf.predict(X) # or predictw
  return predcs 

def evaluate_surv_days(data_in_test_order:pd.DataFrame, gt_col:str, REVERSE_EVAL:bool = False) -> Dict[str, float]:
  """
  data_in_test_order: include "pred" columns 
  """
  maps = {}
  for project, df_proj in data_in_test_order.groupby('project'):
    print (f"For {project}")
    aps = []
    for bid, df in df_proj.groupby('bid'):
      bid = int(bid)
      n_surv = (df.status == 'surv').sum()
      n_others = (df.status != 'surv').sum()
      #
      predcs = df.pred.values 
      pred_ranks = rankdata(-predcs, method = 'ordinal') # here, greater, higher rank 
      gts = df[gt_col].values 
      gt_ranks = rankdata(-gts, method = 'ordinal') # here, greater, higher rank 
      coeff, pval = spearmanr(pred_ranks, gt_ranks)
      print (f"\tFor {bid} ({n_surv} vs {n_others}), coeff: {coeff}, pval: {pval}")

      ps, cnt  = [], 0
      status_vs = df.status.values 
      n_surv = (df.status == 'surv').sum()
      if n_surv == 0: 
        print (f"\tFor {bid}, none surived"); continue 
      elif n_surv == len(df):
        print (f"\tFor {bid}, all survived"); continue 
      
      if not REVERSE_EVAL:
        for status, r in sorted(list(zip(status_vs, pred_ranks)), key = lambda v:v[1]):
          if status == 'surv':
            cnt += 1; ps.append(cnt/r)
      else:
        for i,(status, r) in enumerate(
          sorted(list(zip(status_vs, pred_ranks)), key = lambda v:v[1], reverse=True)): # reverse rank
          if status != 'surv':
            cnt += 1; ps.append(cnt/(i+1))
      # compite AP 
      ap = np.mean(ps)
      print (f"\tFor {bid} ({n_surv} vs {n_others}), AP: {ap}")
      aps.append(ap)
    # compute MAP 
    if len(aps) > 0:
      map = np.mean(aps)
      print (f"Among {project} {len(aps)}, MAP: {map}")
      maps[project] = map
    else:
      maps[project] = None 
  print (f"MAP: {np.mean([v for v in maps.values() if v is not None])}")
  return maps 


def evaluate_reveal_predc(data_in_test_order:pd.DataFrame, prob:bool = True, idx_to_reveal_label:int = 0) -> Dict[str, float]:
  """
  data_in_test_order: include "pred" columns 
  compute 
  """
  from sklearn.metrics import balanced_accuracy_score, accuracy_score
  bal_accs, maps = {}, {}
  for project, df_proj in data_in_test_order.groupby('project'): # for each project
    aps = []
    if prob:
      for bid, df in df_proj.groupby('bid'):      
        bid = int(bid)
        labels = df.label.values 
        predcs = df.pred.values if not prob else np.argmax(np.array([np.array(v) for v in df.pred.values]), axis = 1)
        #if len(set(labels)) == 1:
        #  bal_acc = accuracy_score(labels, predcs)
        #else:
        #  bal_acc = balanced_accuracy_score(labels, predcs)
        #bal_accs_loc.append(bal_acc)
        #print (f"\tFor {bid}, balAcc: {bal_acc}")
        # evaluate whether we can rank interesting (= revealing mutants) at the top 
        #if prob: 
        predcs = np.array([np.array(v) for v in df.pred.values])[:,idx_to_reveal_label]
        n_reveal = (df.status == 'reveal').sum()
        if n_reveal == 0: 
          #print (f"\tFor {bid}, none revealed")
          continue 
        elif n_reveal == len(df):
          #print (f"\tFor {bid}, all revealed")
          continue 
        ps, cnt  = [], 0 
        status_vs = df.status.values 
        for i, (status, _) in enumerate(sorted(zip(status_vs, predcs), key = lambda v:v[1], reverse=True)):
          if status == 'reveal':
            cnt += 1; ps.append(cnt/(i+1))
        # compite AP 
        aps.append(np.mean(ps))
    
    # compute MAP 
    if len(aps) > 0:
      map = np.mean(aps)
      print (f"Among {project} {len(aps)}, MAP: {map}")
      maps[project] = map
    else:
      maps[project] = None 
    #print (f"MAP: {np.mean([v for v in maps.values() if v is not None])} and BalAcc: {np.mean(bal_accs_loc)}")
    print (f"For {project},\n\tMAP: {np.mean([v for v in maps.values() if v is not None])}")
    # combined per project 
    labels = df_proj.label.values 
    predcs = df_proj.pred.values if not prob else np.argmax(
      np.array([np.array(v) for v in df_proj.pred.values]), axis = 1)
    if len(set(labels)) == 1:
      total_bal_acc = accuracy_score(labels, predcs)
    else:
      total_bal_acc = balanced_accuracy_score(labels, predcs)
    bal_accs[project] = total_bal_acc
    print (f"\tbalAcc: {total_bal_acc}")
    #
    if prob: 
      predcs = np.array([np.array(v) for v in df_proj.pred.values])[:,idx_to_reveal_label]
      n_reveal = (df_proj.status == 'reveal').sum()
      if n_reveal == 0: continue
      elif n_reveal == len(df_proj): continue
      ps, cnt  = [], 0 
      status_vs = df_proj.status.values 
      for i, (status, _) in enumerate(sorted(zip(status_vs, predcs), key = lambda v:v[1], reverse=True)):
        if status == 'reveal':
          cnt += 1
          ps.append(cnt/(i+1))
      # compite AP 
      ap = np.mean(ps)
      print (f"\tAP: {ap}")
      maps[project] = ap
  return bal_accs, maps


def evaluate_reveal_predc_balacc(data_in_test_order:pd.DataFrame) -> Dict[str, float]:
  """
  data_in_test_order: include "pred" columns 
  computed per project, b/c in several versions, there is only one type of mutants
  """
  from sklearn.metrics import balanced_accuracy_score, accuracy_score
  bal_accs = {}
  for project, df_proj in data_in_test_order.groupby('project'):
    labels = df_proj.label.values 
    predcs = np.argmax(np.array([np.array(v) for v in df_proj.pred.values]), axis = 1)
    if len(set(labels)) == 1:
      total_bal_acc = accuracy_score(labels, predcs)
    else:
      total_bal_acc = balanced_accuracy_score(labels, predcs)
    bal_accs[project] = total_bal_acc
    print (f"For {project}, balAcc: {total_bal_acc}")
  return bal_accs

def evaluate_reveal_predc_map(data_in_test_order:pd.DataFrame, idx_to_reveal_label:int = 0) -> Dict[str, float]:
  """
  data_in_test_order: include "pred" columns 
  compute 
  """
  maps = {}
  for project, df_proj in data_in_test_order.groupby('project'):
    aps = []
    for bid, df in df_proj.groupby('bid'):      
      bid = int(bid)
      predcs = np.array([np.array(v) for v in df.pred.values])[:,idx_to_reveal_label] # 
      n_reveal = (df.status == 'reveal').sum()
      if n_reveal == 0: 
        continue 
      elif n_reveal == len(df):
        continue 
      ps, cnt  = [], 0 
      status_vs = df.status.values 
      for i, (status, _) in enumerate(sorted(zip(status_vs, predcs), key = lambda v:v[1], reverse=True)):
        if status == 'reveal':
          cnt += 1; ps.append(cnt/(i+1))
      # compite AP 
      aps.append(np.mean(ps))

    # compute MAP 
    if len(aps) > 0:
      maps[project] = np.mean(aps)
    else:
      maps[project] = None 
    print (f"Among {project} {len(aps)}, MAP: {maps[project]}")
    print (f"For {project},\n\tMAP: {np.mean([v for v in maps.values() if v is not None])}")
    predcs = np.array([np.array(v) for v in df_proj.pred.values])[:,idx_to_reveal_label]
    n_reveal = (df_proj.status == 'reveal').sum()
    
    if n_reveal == 0: continue
    elif n_reveal == len(df_proj): continue
    
    ps, cnt  = [], 0 
    status_vs = df_proj.status.values 
    for i, (status, _) in enumerate(sorted(zip(status_vs, predcs), key = lambda v:v[1], reverse=True)):
      if status == 'reveal':
        cnt += 1
        ps.append(cnt/(i+1))
    # compite AP 
    ap = np.mean(ps)
    print (f"\tAP: {ap}") # average precison in total: across different versions but within the same project, can we rank revealing mutants near the top?
    maps[project] = ap
  return maps 

def format_output(output:pd.DataFrame, gt_col:str) -> pd.DataFrame:
  """
  toSave: project, bid, mutK, mutOp, lno, status, pred 
  """
  pred_labels = output.pred.apply(lambda v:np.argmax(np.array(v))).values   
  prob_0 = output.pred.apply(lambda v:v[0]).values 
  prob_1 = output.pred.apply(lambda v:v[1]).values 
  prob_2 = output.pred.apply(lambda v:v[2]).values 
  output['prob_0'] = prob_0
  output['prob_1'] = prob_1
  output['prob_2'] = prob_2
  output['pred_label'] = pred_labels
  #to_save = output[['project', 'bid', 'mutK', 'mutOp', 'lno', 'status', gt_col, 'label', 'pred_label', 'prob_0', 'prob_1', 'prob_2']]
  to_save = output # save all
  return to_save 

def feature_analysis(model, features:List[str], data:pd.DataFrame, top_n:int = 5):
  fimps = model.feature_importances_
  sorted_fimps_and_features = sorted(list(zip(fimps, features)), key = lambda v:v[0], reverse=True)
  from scipy.stats import spearmanr
  top_feature_and_fimp_pairs = sorted_fimps_and_features[:top_n]
  last_top_fimpv = top_feature_and_fimp_pairs[-1][0]
  if sorted_fimps_and_features[top_n][0] < last_top_fimpv:
    pass 
  else:
    pass 
  for i, (fimp, feature) in enumerate(sorted_fimps_and_features[:top_n]):
    # 
    labels = data.label.values.copy()
    # replace other (1,2) as all 1
    #labels[np.where(labels == 2)[0]] = 1 
    labels[np.where(labels == 2)[0]] = 1 
    feature_vs = data[feature].values 
    print (f"top {i+1}", feature, fimp, spearmanr(labels, feature_vs))

def get_top_n_features(model, features:List[str], top_n:int = 5) -> List:
  fimps = model.feature_importances_
  sorted_fimps_and_features = sorted(list(zip(fimps, features)), key = lambda v:v[0], reverse=True)
  top_feature_and_fimp_pairs = sorted_fimps_and_features[:top_n]
  last_top_fimpv = top_feature_and_fimp_pairs[-1][0]
  if sorted_fimps_and_features[top_n][0] < last_top_fimpv:
    return top_feature_and_fimp_pairs
  else:
    ret = sorted_fimps_and_features[:top_n]
    for fimp, feature in sorted_fimps_and_features[top_n:]:
      if fimp == last_top_fimpv:
        ret.append([fimp, feature])
    print (f"Total {len(ret) - top_n} addtionals retrieved")
    return ret 
  
def get_feature_imps(model, features:List[str]) -> List:
  fimps = model.feature_importances_
  return {feature:fimp for fimp, feature in zip(fimps, features)}


def feature_analysis_to_gt(feature_cols:List[str], data:pd.DataFrame, gt_col:str = 'debt_time', excl_status:str = None):
  from scipy.stats import spearmanr
  if excl_status is not None:
    data = data.loc[~(data.status == excl_status)]
  gt_vals = data[gt_col]
  rets = {}
  for feature_col in feature_cols:
    feature_vs = data[feature_col].values 
    rho, pval = spearmanr(gt_vals, feature_vs)
    rets[feature_col] = (rho, pval)
  return rets