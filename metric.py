import bottleneck as bn
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Metrics:

    @staticmethod
    def compute(pred_scores, ground_truth, metrics_list):
        results = {}
        for metric in metrics_list:
            try:
                if "@" in metric:
                    met, k = metric.split("@")
                    met_foo = getattr(Metrics, f"{met.lower()}_at_k")
                    results[metric] = met_foo(pred_scores, ground_truth, int(k))
                else:
                    results[metric] = getattr(Metrics, metric)(pred_scores, ground_truth)
            except AttributeError:
                logger.warning(f"Skipped unknown metric '{metric}'.")
        return results

    @staticmethod
    def ndcg_at_k(pred_scores, ground_truth, k=100):
        n_users = pred_scores.shape[0]
        idx_topk_part = bn.argpartition(-pred_scores, k, axis=1)
        topk_part = pred_scores[np.arange(n_users)[:, np.newaxis], idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(n_users)[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k + 2))
        DCG = (ground_truth[np.arange(n_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(int(n), k)]).sum() for n in ground_truth.sum(axis=1)])
        return DCG / IDCG

    @staticmethod
    def recall_at_k(pred_scores, ground_truth, k=100):
        idx = bn.argpartition(-pred_scores, k, axis=1)
        pred_scores_binary = np.zeros_like(pred_scores, dtype=bool)
        pred_scores_binary[np.arange(pred_scores.shape[0])[:, np.newaxis], idx[:, :k]] = True
        X_true_binary = (ground_truth > 0)
        num = (np.logical_and(X_true_binary, pred_scores_binary).sum(axis=1)).astype(np.float32)
        recall = num / np.minimum(k, X_true_binary.sum(axis=1)) #TODO check this, it seems precision to me
        return recall
