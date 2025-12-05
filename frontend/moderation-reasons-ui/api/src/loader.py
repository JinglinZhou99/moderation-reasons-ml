# moderation-reasons-ui/api/src/loader.py
import os, json, joblib
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

DEFAULT_LABELS = ["violence","sexual","hate"]

class ModelBundle:
    def __init__(self, mock: bool = True, model_dir: str = "models/lr"):
        self.mock = mock
        self.model_dir = Path(model_dir)
        self.labels: List[str] = DEFAULT_LABELS[:]  # 默认顺序
        self.thresholds: Dict[str, float] = {l: 0.5 for l in DEFAULT_LABELS}
        self.vectorizer = None

        # 两种模型格式的容器：
        self.model = None              # OneVsRestClassifier / CalibratedClassifier / 任意有 predict_proba 的估计器
        self.estimators = None         # 新格式的按标签列表: [("lr"|"constant", est_or_prob), ...]

        if not mock:
            self._load_real(self.model_dir)

    def _load_real(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")

        # 读取标签顺序（如存在）
        label_cfg = model_dir / "label_config.json"
        if label_cfg.exists():
            cfg = json.loads(label_cfg.read_text())
            if isinstance(cfg, dict) and "labels" in cfg and isinstance(cfg["labels"], list):
                self.labels = cfg["labels"]

        # 阈值
        thr_path = model_dir / "thresholds.json"
        if thr_path.exists():
            self.thresholds = json.loads(thr_path.read_text())

        # 向量器
        vec_p = model_dir / "vectorizer.pkl"
        if not vec_p.exists():
            raise FileNotFoundError(f"Missing vectorizer.pkl in {model_dir}")
        self.vectorizer = joblib.load(vec_p)

        # 模型
        mdl_p = model_dir / "model.joblib"
        if not mdl_p.exists():
            raise FileNotFoundError(f"Missing model.joblib in {model_dir}")
        artifact = joblib.load(mdl_p)

        # 兼容两种格式
        if hasattr(artifact, "predict_proba"):
            # 传统 OneVsRest / 校准后的分类器
            self.model = artifact
            self.estimators = None
        elif isinstance(artifact, dict) and "estimators" in artifact:
            # 新格式：逐标签估计器或常量
            self.model = None
            self.estimators = artifact["estimators"]
        else:
            raise ValueError(
                "model.joblib 既不是可调用 predict_proba 的估计器，也不是包含 'estimators' 的字典；"
                "请确认与训练端保存格式一致。"
            )

    def predict_probs(self, texts: List[str]) -> np.ndarray:
        """
        返回形状 (N, L) 的概率矩阵，列顺序与 self.labels 一致
        """
        if self.mock:
            return None

        X = self.vectorizer.transform(texts)

        # 老格式：直接用模型的 predict_proba
        if self.model is not None:
            # 一些多标签实现会返回 list[ndarray]，也可能返回 (N,2) 的数组；统一成 (N,L)
            probs_list = self.model.predict_proba(X)
            if isinstance(probs_list, list):
                # 列表里每个元素是 (N,2)，取正类列
                cols = []
                for p in probs_list:
                    p = np.asarray(p)
                    if p.ndim == 2 and p.shape[1] >= 2:
                        cols.append(p[:, 1])
                    else:
                        # 极端情况下退化为一列，直接用该列
                        cols.append(p.ravel())
                P = np.vstack(cols).T
            else:
                # 直接就是 (N, L) 的概率
                P = np.asarray(probs_list)
            return P

        # 新格式：逐标签合成概率
        if self.estimators is not None:
            N = X.shape[0]
            L = len(self.labels)
            P = np.zeros((N, L), dtype=float)
            for j, (kind, est) in enumerate(self.estimators):
                if kind == "constant":
                    P[:, j] = float(est)
                else:
                    proba = est.predict_proba(X)
                    proba = np.asarray(proba)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        P[:, j] = proba[:, 1]
                    else:
                        P[:, j] = proba.ravel()
            return P

        raise RuntimeError("未加载任何模型。请检查 MOCK 和 MODEL_DIR 设置。")

    def label_thresholds(self) -> Dict[str, float]:
        # 确保按照 self.labels 顺序返回
        return {l: float(self.thresholds.get(l, 0.5)) for l in self.labels}

