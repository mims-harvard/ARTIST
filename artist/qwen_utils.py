import os
import json
import pandas as pd
import re
import re, json, torch, numpy as np
from collections import Counter
from typing import Optional, List, Tuple

def extract_answer_letter(llm_response: str) -> str:
    """Extract answer letter from LLM response"""
    response = llm_response.strip()

    # Look for single letters at the beginning
    for char in response:
        if char in 'ABCDEFGHIJ':
            return char

    # Look for explicit patterns
    patterns = [
        r'\bANSWER\s+IS\s+([A-J])\b',
        r'\bOPTION\s+([A-J])\b',
        r'\b([A-J])\s*:\s*',
        r'\b([A-J])\)\s*',
        r'The\s+answer\s+is\s+([A-J])',
        r'I\s+choose\s+([A-J])',
        r'My\s+answer\s+is\s+([A-J])',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return "UNKNOWN"

def save_test_results(output_dir, predictions, accuracy, errors, timestamp, metrics=None):
    """Save test results to files"""
    
    info_file = os.path.join(output_dir, 'run_info.json')
    run_info = {
        'accuracy': accuracy,
        'total_samples': len(predictions),
    }
    
    if metrics:
        run_info['metrics'] = {
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'micro_f1': metrics['micro_f1'],
            'micro_precision': metrics['micro_precision'],
            'micro_recall': metrics['micro_recall']
        }
    
    with open(info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run info to {info_file}")
    
    detailed_file = os.path.join(output_dir, f'predictions_detailed.json')
    with open(detailed_file, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'total_samples': len(predictions),
            'predictions': predictions,
            'metrics': metrics,
        }, f, indent=2, default=str)
    print(f"Saved detailed results to {detailed_file}")
    
    if predictions:
        summary_data = []
        for i, pred in enumerate(predictions):
            summary_data.append({
                'sample_id': i,
                'context': pred['context'],
                'prediction': pred['result'],
                'ground_truth': pred['label'],
                'correct': pred['result'].strip().lower() == pred['label'].strip().lower(),
                'ts_length': len(pred['ts']) if 'ts' in pred else 0
            })
        
        df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f'predictions_summary.csv')
        df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
        
        accuracy_by_label = df.groupby('ground_truth')['correct'].agg(['count', 'sum', 'mean']).round(3)
        accuracy_file = os.path.join(output_dir, f'accuracy_breakdown.csv')
        accuracy_by_label.to_csv(accuracy_file)
        print(f"Saved accuracy breakdown to {accuracy_file}")
        
        if metrics:
            # Save per-class metrics
            per_class_data = []
            for class_name, class_metrics in metrics['per_class'].items():
                per_class_data.append({
                    'class': class_name,
                    'precision': class_metrics['precision'],
                    'recall': class_metrics['recall'],
                    'f1': class_metrics['f1'],
                    'support': class_metrics['support']
                })
            
            per_class_df = pd.DataFrame(per_class_data)
            per_class_file = os.path.join(output_dir, f'per_class_metrics.csv')
            per_class_df.to_csv(per_class_file, index=False)
            print(f"Saved per-class metrics to {per_class_file}")
            
            # Save summary metrics
            summary_metrics = {
                'accuracy': accuracy,
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
                'macro_precision': metrics['macro_precision'],
                'macro_recall': metrics['macro_recall'],
                'micro_f1': metrics['micro_f1'],
                'micro_precision': metrics['micro_precision'],
                'micro_recall': metrics['micro_recall'],
                'total_samples': len(predictions)
            }
            
            summary_file = os.path.join(output_dir, f'summary_metrics.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_metrics, f, indent=2)
            print(f"Saved summary metrics to {summary_file}")
    
    if errors:
        error_file = os.path.join(output_dir, f'errors_{timestamp}.txt')
        with open(error_file, 'w') as f:
            f.write('\n'.join(errors))
        print(f"Saved errors to {error_file}")
    
    print(f"All results saved in {output_dir}/ directory")


def extract_answer(text: str):
    ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    m = ANSWER_RE.search(text);  return m.group(1).strip() if m else None


def extract_yes_no_answer(text: str) -> str:
    """
    Extracts 'yes' or 'no' from a text string.
    Returns the lowercase answer ('yes' or 'no'), or None if not found.
    """
    if not text or not isinstance(text, str):
        return None
    
    # Normalize text
    text = text.strip().lower()
    
    # Common patterns (handles 'Answer: yes', 'answer is no', etc.)
    patterns = [
        r'\banswer[:\s-]*\b(yes|no)\b',  # e.g., "Answer: yes"
        r'\b(is|was|be)\s*(yes|no)\b',   # e.g., "The answer is no"
        r'\b(yes|no)\b'                  # standalone yes/no
    ]
    
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            # Return the yes/no match (usually last group)
            for group in match.groups()[::-1]:
                if group in ['yes', 'no']:
                    return group
    
    return None

def extract_answer_letter_from_text(text: str) -> Optional[str]:
    """
    Extract a multiple-choice answer letter (A/B/C/D) from free-form text.
    Returns 'A'|'B'|'C'|'D' or None if not confidently found.

    Strategy:
      - Search multiple regex patterns capturing a single letter A-D.
      - If several matches occur, return the majority; if tied, return the last.
    """
    _PATTERNS: List[re.Pattern] = [
    # "the answer is: B", "final answer: 'C'", "answer is option D"
    re.compile(
        r"""(?ix)
        \b(?:final\s+answer|correct\s+answer|the\s+answer|answer)\s*(?:is|:)?    # answer lead-in
        \s*(?:["'\s]*(?:option|choice)?\s*)?                                     # optional 'option'
        ([ABCD])\b
        """
    ),
    # "This matches option B", "corresponds to choice C"
    re.compile(
        r"""(?ix)
        \b(?:matches?|corresponds\s+to|is)\s+(?:option|choice)\s*["'\s]*
        ([ABCD])\b
        """
    ),
    # "option (b) is correct", "choice d", "(C) selected"
    re.compile(
        r"""(?ix)
        \b(?:option|choice)\s*[\(\s]*([ABCD])[\)\s]*\b
        (?:is\s+correct|selected|chosen)?                                        # optional tail
        """
    ),
    # "I choose d", "selected (A)", "pick C"
    re.compile(
        r"""(?ix)
        \b(?:i\s*(?:choose|pick|select)|choose|selected|chosen|pick)\s*
        (?:option|choice)?\s*["'\s]*
        ([ABCD])\b
        """
    ),
    # Fallback: "answer: (B)"
    re.compile(
        r"""(?ix)
        \banswer\s*[:\-]\s*[\(\s"']*([ABCD])[\)\s"']*\b
        """
    ),
    ]

    if not text:
        return None

    candidates: List[str] = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            candidates.append(m.group(1).upper())

    if not candidates:
        return None

    # Prefer the majority vote if there are multiple mentions; else last mention.
    counts = Counter(candidates)
    most_common = counts.most_common()
    if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
        return most_common[0][0]
    return candidates[-1]  # tie-breaker: last mention

def extract_answer_letter_from_text_ext(text: str, valid_letters: str = "ABCDEFGHIJKLMNOP") -> Optional[str]:
    """
    Extract a multiple-choice answer letter (A–J) from free-form text.
    Returns one of the valid_letters (default: A–J) or None if not confidently found.

    Strategy:
      - Search multiple regex patterns capturing a single valid letter.
      - If several matches occur, return the majority; if tied, return the last.
    """
    if not text:
        return None

    # Build letter class dynamically (e.g., [A-J])
    letter_class = f"[{valid_letters}]"

    _PATTERNS: List[re.Pattern] = [
        # "the answer is: B", "final answer: 'C'", "answer is option D"
        re.compile(
            rf"""(?ix)
            \b(?:final\s+answer|correct\s+answer|the\s+answer|answer)\s*(?:is|:)?    # lead-in
            \s*(?:["'\s]*(?:option|choice)?\s*)?                                     # optional 'option'
            ({letter_class})\b
            """
        ),
        # "This matches option B", "corresponds to choice C"
        re.compile(
            rf"""(?ix)
            \b(?:matches?|corresponds\s+to|is)\s+(?:option|choice)\s*["'\s]*
            ({letter_class})\b
            """
        ),
        # "option (b) is correct", "choice d", "(C) selected"
        re.compile(
            rf"""(?ix)
            \b(?:option|choice)\s*[\(\s]*({letter_class})[\)\s]*\b
            (?:is\s+correct|selected|chosen)?                                        # optional tail
            """
        ),
        # "I choose d", "selected (A)", "pick C"
        re.compile(
            rf"""(?ix)
            \b(?:i\s*(?:choose|pick|select)|choose|selected|chosen|pick)\s*
            (?:option|choice)?\s*["'\s]*
            ({letter_class})\b
            """
        ),
        # Fallback: "answer: (B)"
        re.compile(
            rf"""(?ix)
            \banswer\s*[:\-]\s*[\(\s"']*({letter_class})[\)\s"']*\b
            """
        ),
    ]

    candidates: List[str] = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            candidates.append(m.group(1).upper())

    if not candidates:
        return None

    # Prefer majority vote if there are multiple mentions; else last mention
    counts = Counter(candidates)
    most_common = counts.most_common()
    if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
        return most_common[0][0]
    return candidates[-1]  # tie-breaker: last mention

def compute_token_f1(pred: str, gold: str):
    """Simple token-level F1 score."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    
    if not gold_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    intersection = pred_tokens & gold_tokens
    
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"f1": f1, "precision": precision, "recall": recall}


def compute_bleu(pred: str, gold: str, max_n: int = 4) -> float:
    """Simple BLEU score computation."""
    from collections import Counter
    import math
    
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(gold_tokens) else math.exp(1 - len(gold_tokens) / len(pred_tokens))
    
    # N-gram precision
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)])
        gold_ngrams = Counter([tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens) - n + 1)])
        
        matches = sum(min(pred_ngrams[ng], gold_ngrams[ng]) for ng in pred_ngrams)
        total = sum(pred_ngrams.values())
        
        precision = matches / total if total > 0 else 0
        if precision > 0:
            precisions.append(precision)
    
    if not precisions:
        return 0.0
    
    # Geometric mean of precisions
    log_precision = sum(math.log(p) for p in precisions) / len(precisions)
    bleu = bp * math.exp(log_precision)
    
    return bleu


def compute_rouge_l(pred: str, gold: str) -> float:
    """Simple ROUGE-L (longest common subsequence) score."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # LCS length computation
    m, n = len(pred_tokens), len(gold_tokens)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
    
    lcs_length = lcs[m][n]
    
    # ROUGE-L F1
    precision = lcs_length / len(pred_tokens) if pred_tokens else 0
    recall = lcs_length / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1

def parse_tool_segs(text: str):
    segs, i = [], 0
    segs_names = []
    while True:
        s = text.find("<tool_call>", i)
        if s == -1: break
        e = text.find("</tool_call>", s)
        if e == -1: break
        raw = text[s+11:e].strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # Attempt to fix unquoted ts_type values (e.g., ts_type: original)
            try:
                fixed = re.sub(r'("ts_type")\s*:\s*([a-zA-Z_][\w]*)\b', r'\1: "\2"', raw)
                parsed = json.loads(fixed)
            except Exception:
                # Fallback: extract via regex
                ts_match = re.search(r'"ts_seg"\s*:\s*\[(\d+)\s*,\s*(\d+)\]', raw)
                name_match = re.search(r'"ts_type"\s*:\s*"?([a-zA-Z_][\w]*)"?', raw)
                if ts_match:
                    start = int(ts_match.group(1))
                    end = int(ts_match.group(2))
                    if end - start <= 8:
                        end = start + 8
                    segs.append([start, end])
                    segs_names.append(name_match.group(1) if name_match else None)
                i = e + 12
                continue
        if parsed is not None:
            args = parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
            ts = args.get("ts_seg", None)
            # ts_type can be at top-level or inside arguments
            seg_name = parsed.get("ts_type", None)
            if seg_name is None:
                seg_name = args.get("ts_type", 'original')
            if isinstance(ts, list) and len(ts) == 2 and all(isinstance(x, (int, float)) for x in ts):
                start = int(ts[0]); end = int(ts[1])
                if end - start <= 8:
                    end = start + 8
                segs.append([start, end])
                segs_names.append(seg_name)
        i = e + 12
    return segs, segs_names

def calculate_comprehensive_metrics(predictions):
        """
        Calculate comprehensive metrics including precision, recall, F1-score, and confusion matrix
        """
        from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
        import numpy as np
        
        if not predictions:
            return {
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'micro_f1': 0.0,
                'micro_precision': 0.0,
                'micro_recall': 0.0,
                'per_class': {},
                'confusion_matrix': None,
                'classification_report': None
            }
        
        # Extract true labels and predictions
        y_true = [pred['label'].strip().lower() for pred in predictions]
        y_pred = [pred['result'].strip().lower() for pred in predictions]
        
        # Get unique classes
        unique_classes = sorted(list(set(y_true + y_pred)))
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=unique_classes, zero_division=0
        )
        
        # Calculate macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, labels=unique_classes, 
                                    output_dict=True, zero_division=0)
        
        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, class_name in enumerate(unique_classes):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        return {
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'micro_f1': float(micro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'per_class': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'unique_classes': unique_classes
        }