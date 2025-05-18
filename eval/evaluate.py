"""
功能：评估函数
"""
import jieba
import sacrebleu  
from rouge_score import rouge_scorer
from rouge_chinese import Rouge
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 标点过滤
def remove_punctuation(text):
    punc = '！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛"”„‟…‧﹏' + string.punctuation
    return re.sub(r'[{}]+'.format(re.escape(punc)), '', text)


# 计算 BLEU-1 到 BLEU-4 分数
def calculate_bleu_scores(references, generated):
    """
    计算 BLEU-1 到 BLEU-4 分数
    - `param` `reference_tokens`: 参考文本 (列表格式，包含多个参考句子)
    - `param` `generated_tokens`: 生成文本 (字符串格式)
    :return: BLEU-1 到 BLEU-4 分数
    """
    generated = ' '.join(jieba.cut(generated.replace(" ", "")))
    references = [[' '.join(jieba.cut(ref.replace(" ", ""))) for ref in ref_group] 
                  for ref_group in references]

    sacre_bleu = sacrebleu.corpus_bleu([generated], references, tokenize="zh")
    bleu1 = sacre_bleu.precisions[0] / 100
    bleu2 = sacre_bleu.precisions[1] / 100
    bleu3 = sacre_bleu.precisions[2] / 100
    bleu4 = sacre_bleu.precisions[3] / 100
    return bleu1, bleu2, bleu3, bleu4
    
    """
    # 使用 nltk 计算 BLEU 分数
    smoothing = SmoothingFunction().method1
    weights = [
        (1, 0, 0, 0),    # BLEU-1
        (0.5, 0.5, 0, 0), # BLEU-2
        (0.33, 0.33, 0.33, 0), # BLEU-3
        (0.25, 0.25, 0.25, 0.25) # BLEU-4
    ]
    bleu_scores = [
        sentence_bleu(references[0], generated, weights=w, smoothing_function=smoothing)
        for w in weights
    ]
    return bleu_scores
    """

# 计算 ROUGE 分数
def calculate_rouge_scores(reference, generated):
    """
    计算 ROUGE-1, ROUGE-2, ROUGE-L 分数
    - `param` `reference`: 参考文本 (字符串格式)
    - `param` `generated`: 生成文本 (字符串格式)
    :return: 各类 ROUGE 分数
    """
    reference_intensive = " ".join(jieba.cut(reference.replace(" ", "")))
    generated_intensive = " ".join(jieba.cut(generated.replace(" ", "")))
    rouge = Rouge()
    scores = rouge.get_scores(generated_intensive, reference_intensive)
    scores = scores[0]
    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']


# 计算 BLEURT 分数
def calculate_bleurt_score(references, candidates, tokenizer, model, device):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt').to(device)
        scores = model(**inputs).logits.flatten().tolist()
    
    return scores


# 计算错误类型联合准确度（完全匹配才算对）
def calculate_error_type_accuracy(true_error_types, predicted_error_types):
    """
    单样本联合准确度（两个集合完全一致为正确）
    """
    true_set = set(true_error_types or [])
    pred_set = set(predicted_error_types or [])
    return 1.0 if true_set == pred_set else 0.0


# 计算 Acc-1（只要有一个匹配就算对；参考为空视为正确）
def calculate_acc_1(true_error_types, predicted_error_types):
    """
    单样本 Acc-1 准确率：参考为空则视为正确；否则只要有交集就正确。
    """
    true_set = set(true_error_types or [])
    pred_set = set(predicted_error_types or [])
    if not true_set:
        return 1.0
    return 1.0 if true_set & pred_set else 0.0



# 计算所有指标
def calculate_all_metrics(reference, generated, tokenizer, model, ref_error_types, pred_error_types, device):
    """
    计算 BLEU-1 到 BLEU-4, ROUGE-1 到 ROUGE-L, BLEURT 分数，以及准确度
    - `param` `tokenizer`: 分词器
    - `param` `reference`: 参考文本 (列表格式，包含多个参考句子)
    - `param` `generated`: 生成文本 (字符串格式)
    - `param` `ref_error_types`: 真实错误类型
    - `param` `pred_error_types`: 预测错误类型
    :return: 各类指标的分数
    """
    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores([[reference]], generated)
    rouge_1, rouge_2, rouge_l = calculate_rouge_scores(reference, generated)
    bleurt_scores = calculate_bleurt_score([reference], [generated], tokenizer, model, device)

    # 计算错误类型准确度
    error_type_accuracy = calculate_error_type_accuracy(ref_error_types, pred_error_types)
    
    # 计算 Acc-1
    acc_1 = calculate_acc_1(ref_error_types, pred_error_types)

    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l,
        'BLEURT': np.mean(bleurt_scores),  
        'Joint Accuracy': error_type_accuracy,
        'Acc-1': acc_1
    }


rouge_scorer_en = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
def calculate_all_metrics_en(reference, generated, tokenizer, model, device):
    """
    计算英文 BLEU、ROUGE、BLEURT
    """
    # 使用 sacrebleu 计算 BLEU 分数
    references = [reference]  # 单个参考文本的列表
    generated_text = generated  # 生成文本

    sacre_bleu = sacrebleu.corpus_bleu([generated], [references], tokenize="intl")  # 使用 sacrebleu
    bleu1 = sacre_bleu.precisions[0] / 100
    bleu2 = sacre_bleu.precisions[1] / 100
    bleu3 = sacre_bleu.precisions[2] / 100
    bleu4 = sacre_bleu.precisions[3] / 100

    # ROUGE 分数
    rouge_scorer_en = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_en.score(reference, generated)

    # BLEURT 分数
    bleurt_scores = calculate_bleurt_score([reference], [generated], tokenizer, model, device)

    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'ROUGE-1': rouge_scores["rouge1"].fmeasure,
        'ROUGE-2': rouge_scores["rouge2"].fmeasure,
        'ROUGE-L': rouge_scores["rougeL"].fmeasure,
        'BLEURT': np.mean(bleurt_scores),
    }