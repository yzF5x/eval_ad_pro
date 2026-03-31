import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve,roc_auc_score, jaccard_score,average_precision_score,balanced_accuracy_score
import json
import requests
import re
import math
import pandas as pd
import os
def send2api(prediction , prompt = "" , model = "qwen/qwen3-235b-a22b-2507"):
    """
        无法直接提取答案时 调api 让api模型根据测试模型的文本输出得到答案
        prediction是测试模型的文本输出
        prompt是给API model的问题
        发送给API的内容是 prompt + prediction
        可以参考的prompt:
        prompt = "Determine whether there is an anomaly or defect by the semantics of the following paragraph.  If yes, answer \"yes\", otherwise answer \"no\".  No other words are allowed.  No punctuation is allowed. The paragraph is : "
    """
    prompt = "Determine whether there is an anomaly or defect by the semantics of the following paragraph.  If yes, answer \"yes\", otherwise answer \"no\".  No other words are allowed . No punctuation is allowed. The paragraph is : "
    url = "https://openrouter.ai/api/v1/chat/completions"
    ssh_key = ""
    headers= {
            "Authorization": f"Bearer " + ssh_key
    }
    text = {
        "type": "text",
        "text": prompt + f" '{prediction}' "
    }
    content = [text]
    payload = {
        "model": model, 
        "messages": [
          {
            "role": "user",
            "content": content
          }
        ]
    }
    response = requests.post(url = url, headers=headers, json=payload)
    if response.status_code == 200:
        json_llm_answer = response.json()
        choices = json_llm_answer.get('choices',[])
        d = choices[0]
        messages = d.get('message',{})
        content = messages.get('content','')
        # print(prediction,"\n" ,content,"\n")
        return content
    else:
        return "Something Wrong with API"

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
    # print(f"tn = {tn}   tp  = {tp}   fp = {fp}   fn = {fn}   correct_count = {correct_count}")
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    specificity = tn / (tn + fp) 
    mcc = (tp * tn - fp * fn) / math.sqrt ((tp + fp) * (tp + fn) * (tn +fp) * (tn + fn))
    f1_score = 2 * (precision * recall) / (precision + recall) 
    return acc , f1_score, precision, recall , mcc ,balanced_acc , auroc , aupr

def compute_classify_matrics(anomaly_dct):
    # 异常判断的指标
    anomaly_metrics = dict(split = [],Acc = [],F1=[],Precision = [],Recall = [],AUROC=[],AUPR=[])
    for k , v in anomaly_dct.items():
        acc , f1_score, precision, recall , mcc ,balanced_acc , auroc , aupr = calc_binary_classification_metrics(v['true'], v['pred'])
        anomaly_metrics['split'].append(k)
        anomaly_metrics['Acc'].append(acc)
        anomaly_metrics['F1'].append(f1_score )
        anomaly_metrics['Precision'].append(precision )
        anomaly_metrics['Recall'].append(recall )
        anomaly_metrics['AUROC'].append(auroc)
        anomaly_metrics['AUPR'].append(aupr)
    anomaly_score = pd.DataFrame(anomaly_metrics)
    new_row = {}
    for column in anomaly_score.columns:
        if column == "split":  
            new_row[column] = "Mean"
        elif pd.api.types.is_numeric_dtype(anomaly_score[column]):  # 判断是否为数值列
            new_row[column] = anomaly_score[column].mean() 
    anomaly_score.loc[len(anomaly_score)] = new_row
    return anomaly_score


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    # parser.add_argument('--orig_result_path', '-o', type=str, default='/gpfsdata/home/yizhou/Project/VLM/VLM-AD/eval/results/qwen3/MVTecAD_seg_0shot/Qwen3-VL-8B-Instruct/result.json')
    parser.add_argument('--dataset', '-dd', default='mvtec')
    parser.add_argument('--model_path', '-m', default='/home/yizhou/LVLM/merged_model/1015grpo/Qwen2.5-VL-7B-zero-420-GRPO-1010-with-tag-ckpt4800',help='')
    parser.add_argument('--per_head', '-ph', action='store_true')
    parser.add_argument('--with_tag', action='store_true')
    args = parser.parse_args()
    if args.per_head:
        if args.dataset == 'mvtec':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/MVTecAD_seg_0shot' 
        elif args.dataset == 'mpdd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/MPDD_seg_0shot'
        elif args.dataset == 'dagm':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/DAGM_seg_0shot'
        elif args.dataset == 'sdd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/SDD_seg_0shot'
        elif args.dataset =='btad':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/BTAD_seg_0shot'
        elif args.dataset =='dtd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/DTD_seg_0shot'
        elif args.dataset =='wfdd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/per-head/WFDD_seg_0shot'
    else:
        if args.dataset == 'mvtec':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/MVTecAD_seg_0shot' 
        elif args.dataset == 'mpdd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/MPDD_seg_0shot'
        elif args.dataset == 'dagm':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/DAGM_seg_0shot'
        elif args.dataset == 'sdd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/SDD_seg_0shot'
        elif args.dataset =='btad':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/BTAD_seg_0shot'
        elif args.dataset =='dtd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/DTD_seg_0shot'
        elif args.dataset =='wfdd':
            args.save_dir = '/share/home/yizhou_lustre/LVLM-results/attention-outputs/WFDD_seg_0shot'
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if args.with_tag:
        model_name += 'with-tag'
    args.orig_result_path = os.path.join(args.save_dir, model_name,'results/result.json')
    orig_result_path = args.orig_result_path
    with open(orig_result_path,'r') as f:
        results = json.load(f)
    anomaly_dct = {}
    idx = 0
    for k , v in results.items():
        if not anomaly_dct.get(v['category']):
            anomaly_dct[v['category']] = {"pred":[],"true":[]}
        if "nothinking" in orig_result_path:
            pred_yes_or_no = v["pred_reasoning"].lower().replace("addcriterion","")
            pred_yes_or_no = "no" if 'no' in pred_yes_or_no else "yes"
            pred_answer = 1 if 'yes' in pred_yes_or_no else 0
        elif "glm" in orig_result_path.lower():
            # 处理GLM模型的输出结果
            try:
                pred_content_match = re.search(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', v["pred_reasoning"].lower(), re.DOTALL)
                pred_yes_or_no = pred_content_match.group(1).strip() 
                pred_answer = 1 if 'yes' in pred_yes_or_no else 0
                # print(vlmanswer)
            except:
                pred_yes_or_no = send2api(v["pred_reasoning"]).strip()
                idx+=1
                pred_answer = 1 if 'yes' in pred_yes_or_no else 0
                # print(v["id"] , v["pred_reasoning"])
                print(pred_yes_or_no)
        else:
            try:
                pred_content_match = re.search(r'<answer>(.*?)</answer>', v["pred_reasoning"].lower(), re.DOTALL)
                pred_yes_or_no = pred_content_match.group(1).strip() 
                pred_answer = 1 if 'yes' in pred_yes_or_no else 0
                # print(vlmanswer)
            except:
                pred_yes_or_no = send2api(v["pred_reasoning"]).strip()
                idx+=1
                pred_answer = 1 if 'yes' in pred_yes_or_no else 0
                # print(v["id"] , v["pred_reasoning"])
                print(pred_yes_or_no)
        
        # gt_content_match = re.search(r'<answer>(.*?)</answer>', v["gt_reasoning"].lower(), re.DOTALL)
        if args.dataset == 'btad':
            if 'ok' in v['id']:
                gt_answer = 0
            else:
                gt_answer = 1
        else:
            if 'good' in v['id']:
                gt_answer = 0
            else:
                gt_answer = 1
        
        # gt_yes_or_no = gt_content_match.group(1).strip() 
        # gt_answer_ = 1 if 'yes' in gt_yes_or_no else 0
        # if gt_answer != gt_answer_:
        #     print(v["gt_reasoning"])
        #     print(v["id"])
        #     print(gt_answer)
        #     print(gt_answer_)
        anomaly_dct[v['category']]["pred"].append(pred_answer)
        anomaly_dct[v['category']]["true"].append(gt_answer)
        
    anomaly_scores = compute_classify_matrics(anomaly_dct)
    path = os.path.join(orig_result_path.rsplit("/",1)[0],"anomaly_score.xlsx")
    anomaly_scores.to_excel(path,index = False,float_format='%.3f')
    print(idx)