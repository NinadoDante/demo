from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 参考句（分词后的列表）
reference = [['今天', '阳光', '不错']]
candidate = ['今天', '阳光']

smooth = SmoothingFunction().method4
bleu = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
print(f"BLEU-2: {bleu:.4f}")