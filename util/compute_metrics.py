import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve,roc_auc_score, jaccard_score,average_precision_score,balanced_accuracy_score
import pandas as pd
import math
import os
import statistics
def compute_acc(gt,pred,threshold):
    pred_binary = (pred > threshold).astype(int)     
    correct_cnt = (pred_binary == gt).sum()
    return correct_cnt / len(gt)

def compute_seg_metrics(dct):
        #     seg_metrics["threshold(f1max)"].append(threshold_f1max)
        # seg_metrics["acc(f1max)"].append(acc_f1max)
        # seg_metrics["f1_max"].append(f1_max)
        # seg_metrics["precision(f1max)"].append(precision_f1max)
        # seg_metrics["recall(f1max)"].append(recall_f1max)
        
        # seg_metrics["threshold(min_dis)"].append(threshold_min_dis)
        # seg_metrics["acc(min_dis)"].append(acc_min_dis)
        # seg_metrics["f1(min_dis)"].append(f1_min_dis)
        # seg_metrics["precision(min_dis)"].append(precision_min_dis)
        # seg_metrics["recall(min_dis)"].append(recall_min_dis)
        
        # seg_metrics["all_precisions"].append(all_precisions)
        # seg_metrics["all_recalls"].append(all_recalls)
        # seg_metrics["all_thresholds"].append(all_thresholds)
    seg_metrics = {"split" : [],
                    "auroc" : [],
                    "ap" : [],
                    "threshold(f1max)" : [],
                    "acc(f1max)":[],
                    "f1_max" : [],
                    "precision(f1max)" : [],
                    "recall(f1max)" : [],
                    "threshold(min_dis)" : [],
                    "acc(min_dis)" : [],
                    "f1(min_dis)" : [],
                    "precision(min_dis)" : [],
                    "recall(min_dis)" : []
                    }
    for category , info in dct.items():
        # pred = np.array(info["pred"]).flatten()
        # gt = np.array(info["true"]).flatten()
        pred_list = []
        gt_list = []
        for p, g in zip(info["pred"], info["true"]):
            p = np.array(p).flatten()
            g = np.array(g).flatten()
            if len(p) != len(g):
                raise ValueError(f"Length mismatch: pred {len(p)} vs gt {len(g)}")
            pred_list.append(p)
            gt_list.append(g)
        # 拼接所有像素
        pred = np.concatenate(pred_list)
        gt = np.concatenate(gt_list)
        auroc_pixel = roc_auc_score(gt,pred)
        ap_pixel = average_precision_score(gt, pred)
        all_precisions, all_recalls, all_thresholds = precision_recall_curve(gt, pred)
        f1_scores_pixel = (2 * all_precisions * all_recalls) /(all_precisions + all_recalls)
        rms_f1_scores = f1_scores_pixel[np.isfinite(f1_scores_pixel)]
        f1_max = np.max(rms_f1_scores)
        f1_max_index = np.argmax(rms_f1_scores)
        
        precision_f1max = all_precisions[f1_max_index]
        recall_f1max = all_recalls[f1_max_index]
        threshold_f1max = all_thresholds[f1_max_index]
        acc_f1max = compute_acc(gt,pred,threshold_f1max)
        
        distance = (1 - all_recalls)**2 + (1 - all_precisions)**2
        min_dis_idx = np.argmin(distance)
        f1_min_dis = f1_scores_pixel[min_dis_idx]
        precision_min_dis = all_precisions[min_dis_idx]
        recall_min_dis = all_recalls[min_dis_idx]
        threshold_min_dis = all_thresholds[min_dis_idx]       
        acc_min_dis = compute_acc(gt,pred,threshold_min_dis)
        
        
        seg_metrics["split"].append(category)
        seg_metrics["auroc"].append(auroc_pixel)
        seg_metrics["ap"].append(ap_pixel)
        
        seg_metrics["threshold(f1max)"].append(threshold_f1max)
        seg_metrics["acc(f1max)"].append(acc_f1max)
        seg_metrics["f1_max"].append(f1_max)
        seg_metrics["precision(f1max)"].append(precision_f1max)
        seg_metrics["recall(f1max)"].append(recall_f1max)
        
        seg_metrics["threshold(min_dis)"].append(threshold_min_dis)
        seg_metrics["acc(min_dis)"].append(acc_min_dis)
        seg_metrics["f1(min_dis)"].append(f1_min_dis)
        seg_metrics["precision(min_dis)"].append(precision_min_dis)
        seg_metrics["recall(min_dis)"].append(recall_min_dis)
        
    seg_metrics = pd.DataFrame(seg_metrics)
    new_row = {}
    for column in seg_metrics.columns:
        if column == "split":  
            new_row[column] = "Mean"
        elif pd.api.types.is_numeric_dtype(seg_metrics[column]):  # 判断是否为数值列
            new_row[column] = seg_metrics[seg_metrics['split'] != 'overall'][column].mean()
    seg_metrics.loc[len(seg_metrics)] = new_row
    print(seg_metrics)
    return seg_metrics

def calc_binary_classification_metrics(y_true, y_pred):
    """
    y_true和y_pred都是只包含01的列表
    """
    correct_count = sum([1 if y_pred[i] == y_true[i] else 0 for i in range(len(y_pred))])
    total_count = len(y_pred)
    acc = correct_count/total_count 
    auroc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    # 平衡准确率 = (recall + specificity) / 2
    balanced_acc = balanced_accuracy_score(y_true , y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = sum((y_true == 0) & (y_pred == 0))
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    print(f"tn = {tn}   tp  = {tp}   fp = {fp}   fn = {fn}   correct_count = {correct_count}")
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    specificity = tn / (tn + fp) 
    mcc = (tp * tn - fp * fn) / math.sqrt ((tp + fp) * (tp + fn) * (tn +fp) * (tn + fn))
    f1_score = 2 * (precision * recall) / (precision + recall) 
    return acc , f1_score, precision, recall , mcc ,balanced_acc , auroc , aupr

def compute_classify_matrics(anomaly_dct):
    # 异常判断的指标
    anomaly_metrics = dict(split = [],Acc = [],F1=[],Precision = [],Recall = [],Mcc = [],BalancedAcc=[],AUROC=[],AUPR=[])
    for k , v in anomaly_dct.items():
        print(v['true'] , v['pred'])
        acc , f1_score, precision, recall , mcc ,balanced_acc , auroc , aupr = calc_binary_classification_metrics(v['true'], v['pred'])
        anomaly_metrics['split'].append(k)
        anomaly_metrics['Acc'].append(acc)
        anomaly_metrics['F1'].append(f1_score )
        anomaly_metrics['Precision'].append(precision )
        anomaly_metrics['Recall'].append(recall )
        anomaly_metrics['Mcc'].append(mcc )
        anomaly_metrics['BalancedAcc'].append(balanced_acc)
        anomaly_metrics['AUROC'].append(auroc)
        anomaly_metrics['AUPR'].append(aupr)
    anomaly_score = pd.DataFrame(anomaly_metrics)
    new_row = {}
    for column in anomaly_score.columns:
        if column == "split":  
            new_row[column] = "Overall"
        elif pd.api.types.is_numeric_dtype(anomaly_score[column]):  # 判断是否为数值列
            new_row[column] = anomaly_score[column].mean() 
    anomaly_score.loc[len(anomaly_score)] = new_row
    return anomaly_score

    # if not anomaly_dct.get(category):
    #     seg_dct[category] = dict(IoU = [],
    #                             Dice = [],
    #                             Pixel_Accuracy = [],
    #                             Precision = [],
    #                             Recall = [],
    #                             F1_Score = [],
    #                             AUROC = []
    #                             )
    #     anomaly_dct[category] = dict(true = [], pred = [])
    # anomaly_dct[category]["true"].append(gt_answer)
    # anomaly_dct[category]["pred"].append(pred_answer)

    # seg_dct[category]["IoU"].append(sample_seg_metrics["IoU"])
    # seg_dct[category]["Dice"].append(sample_seg_metrics["Dice"])
    # seg_dct[category]["Pixel_Accuracy"].append(sample_seg_metrics["Pixel_Accuracy"])
    # seg_dct[category]["Precision"].append(sample_seg_metrics["Precision"])
    # seg_dct[category]["Recall"].append(sample_seg_metrics["Recall"])
    # seg_dct[category]["F1_Score"].append(sample_seg_metrics["F1_Score"])
    # seg_dct[category]["AUROC"].append(sample_seg_metrics["AUROC"])

    # seg_metrics = dict(split = [],IoU = [],Dice=[],Pixel_Accuracy = [],Precision = [],Recall = [],F1_Score=[],AUROC=[])
    # for category in seg_dct.keys():
    #     seg_metrics[category] = {
    #         "IoU": np.mean(seg_dct[category]["IoU"]),
    #         "Dice": np.mean(seg_dct[category]["Dice"]),
    #         "Pixel_Accuracy": np.mean(seg_dct[category]["Pixel_Accuracy"]),
    #         "Precision": np.mean(seg_dct[category]["Precision"]),
    #         "Recall": np.mean(seg_dct[category]["Recall"]),
    #         "F1_Score": np.mean(seg_dct[category]["F1_Score"]),
    #         "AUROC": np.mean(seg_dct[category]["AUROC"])
    #         }
    # seg_metrics = pd.DataFrame(seg_metrics)
    # for column in seg_metrics.columns:
    #     if column == "split":  
    #         new_row[column] = "Overall"
    #     elif pd.api.types.is_numeric_dtype(seg_metrics[column]):  # 判断是否为数值列
    #         new_row[column] = seg_metrics[column].mean() 
    # seg_metrics.loc[len(seg_metrics)] = seg_metrics
    
    # classify_matrics = compute_classify_matrics(anomaly_dct)