from __future__ import annotations

import math
from collections import Counter
from typing import Any


LEVEL_ORDER = {
    "good": 0,
    "normal": 0,
    "improving": 0,
    "watch": 1,
    "warning": 2,
    "critical": 3,
}

LEVEL_TONES = {
    "good": "good",
    "normal": "good",
    "improving": "good",
    "watch": "warn",
    "warning": "warn",
    "critical": "danger",
}

DANGER_LABEL_KEYWORDS = (
    "danger",
    "warning",
    "violence",
    "violent",
    "threat",
    "fight",
    "assault",
    "attack",
    "abduction",
    "kidnap",
    "collapse",
    "fall",
    "swoon",
    "위험",
    "폭행",
    "위협",
    "납치",
    "쓰러",
)

NORMAL_LABEL_KEYWORDS = ("normal", "safe", "negative", "일상", "정상")


def interpret_training_results(
    metrics: dict | None,
    history: list[dict] | None = None,
    class_report: list[dict] | None = None,
    confusion_matrix: list | None = None,
    data_stats: dict | None = None,
) -> dict:
    """Return rule-based dashboard insights for the current training result.

    The function is intentionally defensive: every input is optional, invalid
    numbers are ignored, and missing reports simply omit that diagnosis group.
    """

    metrics = metrics if isinstance(metrics, dict) else {}
    data_stats = data_stats if isinstance(data_stats, dict) else {}
    history_rows = _normalize_history(history if history is not None else metrics.get("history"))
    labels = _normalize_labels(
        metrics.get("labels")
        or data_stats.get("labels")
        or _labels_from_class_report(class_report)
    )
    final_validation = metrics.get("final_validation") if isinstance(metrics.get("final_validation"), dict) else {}
    latest = metrics.get("latest") if isinstance(metrics.get("latest"), dict) else {}
    best_validation = metrics.get("best_validation") if isinstance(metrics.get("best_validation"), dict) else {}
    class_rows = _normalize_class_report(
        class_report
        if class_report is not None
        else final_validation.get("per_class") or metrics.get("per_class"),
        labels=labels,
    )
    matrix = _normalize_matrix(
        confusion_matrix
        if confusion_matrix is not None
        else final_validation.get("confusion_matrix") or metrics.get("confusion_matrix")
    )
    if matrix and not labels:
        labels = [str(index) for index in range(len(matrix))]

    result: dict[str, Any] = {
        "summary": [],
        "risk_badges": [],
        "diagnostics": [],
        "metric_notes": [],
        "class_insights": [],
        "confusion_insights": [],
        "data_quality_insights": [],
        "trend_insights": [],
        "recommendations": [],
        "overall_level": "good",
        "overall_tone": "good",
    }
    recommendation_keys: set[str] = set()

    def add(bucket: str, level: str, title: str, message: str, **extra: Any) -> None:
        item = {
            "level": _normalize_level(level),
            "tone": _level_tone(level),
            "title": title,
            "message": message,
        }
        item.update(extra)
        result.setdefault(bucket, []).append(item)

    def badge(label: str, level: str, detail: str = "") -> None:
        result["risk_badges"].append(
            {
                "label": label,
                "level": _normalize_level(level),
                "tone": _level_tone(level),
                "detail": detail,
            }
        )

    def recommend(key: str, level: str, message: str) -> None:
        if key in recommendation_keys:
            return
        recommendation_keys.add(key)
        add("recommendations", level, "Next action", message, key=key)

    train_loss = _metric(metrics, latest, final_validation, best_validation, keys=("train_loss",))
    val_loss = _metric(metrics, final_validation, best_validation, latest, keys=("val_loss", "loss"))
    val_cross_entropy = _metric(
        metrics,
        final_validation,
        best_validation,
        latest,
        keys=("val_cross_entropy_loss", "cross_entropy_loss"),
    )
    val_accuracy = _metric(metrics, final_validation, best_validation, latest, keys=("val_accuracy", "accuracy"))
    val_macro_f1 = _metric(metrics, final_validation, best_validation, latest, keys=("val_macro_f1", "macro_f1"))
    train_accuracy = _metric(metrics, latest, keys=("train_accuracy", "accuracy_train"))
    train_macro_f1 = _metric(metrics, latest, keys=("train_macro_f1", "macro_f1_train"))
    balanced_accuracy = _metric(
        metrics,
        final_validation,
        best_validation,
        latest,
        keys=("val_balanced_accuracy", "balanced_accuracy"),
    )
    supported_macro_f1 = _metric(
        metrics,
        final_validation,
        best_validation,
        latest,
        keys=("val_macro_f1_supported", "macro_f1_supported", "supported_macro_f1"),
    )
    mean_pred_confidence = _metric(
        metrics,
        final_validation,
        best_validation,
        latest,
        keys=("mean_pred_confidence", "val_mean_pred_confidence"),
    )
    mean_true_confidence = _metric(
        metrics,
        final_validation,
        best_validation,
        latest,
        keys=("mean_true_confidence", "val_mean_true_confidence"),
    )

    _add_metric_notes(
        add,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        train_loss=train_loss,
        val_loss=val_loss,
        skipped_samples=_skip_total(data_stats),
    )

    overfit_level = "good"
    if train_loss is not None and val_loss is not None:
        loss_gap = val_loss - train_loss
        loss_ratio = val_loss / max(train_loss, 1e-6)
        if train_loss <= 0.35 and val_loss >= 1.5 and loss_ratio >= 3.0:
            overfit_level = "critical" if val_loss >= 2.5 or loss_ratio >= 8.0 else "warning"
            add(
                "diagnostics",
                overfit_level,
                "과적합 진단",
                "train loss가 낮은 반면 validation loss가 높아 학습 데이터에 과하게 맞춰졌을 가능성이 큽니다.",
                details={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "loss_gap": round(loss_gap, 6),
                    "loss_ratio": round(loss_ratio, 3),
                },
            )
            recommend("overfit_regularization", "warning", "dropout, weight_decay, label smoothing, augmentation 강화를 함께 검토하세요.")
            recommend("overfit_early_stopping", "watch", "best epoch 이후 validation 성능이 떨어지는지 확인하고 early stopping 기준을 macro F1 중심으로 유지하세요.")
            recommend("overfit_lr", "watch", "validation loss가 계속 높으면 learning rate를 낮추거나 scheduler patience를 줄여보세요.")
    if train_accuracy is not None and val_accuracy is not None and train_accuracy - val_accuracy >= 0.2:
        overfit_level = _max_level(overfit_level, "warning")
        add(
            "diagnostics",
            "warning",
            "정확도 격차",
            "train accuracy와 validation accuracy 차이가 커서 일반화 격차가 의심됩니다.",
            details={"train_accuracy": train_accuracy, "val_accuracy": val_accuracy},
        )
        recommend("more_validation_data", "watch", "validation 분포가 실제 운영 데이터와 맞는지 확인하고 필요한 경우 데이터를 보강하세요.")
    if train_macro_f1 is not None and val_macro_f1 is not None and train_macro_f1 - val_macro_f1 >= 0.2:
        overfit_level = _max_level(overfit_level, "warning")
        add(
            "diagnostics",
            "warning",
            "macro F1 격차",
            "train macro F1에 비해 validation macro F1이 낮아 특정 클래스 일반화가 약할 수 있습니다.",
            details={"train_macro_f1": train_macro_f1, "val_macro_f1": val_macro_f1},
        )
        recommend("class_generalization", "watch", "class별 recall과 misclassified samples를 확인해 라벨 기준이 흔들리는 클래스를 먼저 정리하세요.")

    performance_level = "good"
    if val_accuracy is not None and val_accuracy < 0.5:
        performance_level = _max_level(performance_level, "warning")
        add(
            "diagnostics",
            "warning",
            "전체 정확도 부족",
            "validation accuracy가 0.5 미만이라 전체 예측 안정성이 아직 낮습니다.",
            details={"val_accuracy": val_accuracy},
        )
    if val_macro_f1 is not None and val_macro_f1 < 0.4:
        level = "critical" if val_macro_f1 < 0.25 else "warning"
        performance_level = _max_level(performance_level, level)
        add(
            "diagnostics",
            level,
            "macro F1 부족",
            "validation macro F1이 낮아 일부 클래스가 거의 맞지 않거나 클래스 불균형 영향을 받고 있을 수 있습니다.",
            details={"val_macro_f1": val_macro_f1},
        )
        recommend("macro_f1_class_weight", "warning", _minority_signal_recommendation(metrics))
        recommend("macro_f1_confusion", "watch", "confusion matrix에서 recall이 낮은 클래스와 자주 섞이는 클래스 쌍을 먼저 확인하세요.")
    if balanced_accuracy is not None and balanced_accuracy < 0.5:
        performance_level = _max_level(performance_level, "warning")
        add(
            "diagnostics",
            "warning",
            "balanced accuracy 부족",
            "balanced accuracy가 낮아 클래스별 평균 recall이 충분하지 않습니다.",
            details={"balanced_accuracy": balanced_accuracy},
        )
    if supported_macro_f1 is not None and val_macro_f1 is not None and supported_macro_f1 - val_macro_f1 >= 0.1:
        performance_level = _max_level(performance_level, "watch")
        add(
            "diagnostics",
            "watch",
            "빈/희소 클래스 영향",
            "support가 있는 클래스만 평균한 F1이 전체 macro F1보다 높아 빈 클래스 또는 극소수 클래스가 점수를 끌어내리고 있습니다.",
            details={"macro_f1": val_macro_f1, "supported_macro_f1": supported_macro_f1},
        )

    imbalance_level = "good"
    if val_accuracy is not None and val_macro_f1 is not None and val_accuracy - val_macro_f1 >= 0.2:
        imbalance_level = "warning"
        add(
            "diagnostics",
            "warning",
            "accuracy와 macro F1 괴리",
            "accuracy보다 macro F1이 낮아 다수 클래스 위주로 맞추고 있을 가능성이 있습니다.",
            details={"val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1},
        )
        recommend("imbalance_sampling", "warning", "소수 클래스 oversampling 또는 minority class augmentation을 적용해 클래스별 recall을 끌어올리세요.")

    distribution_findings = _distribution_findings(data_stats)
    for finding in distribution_findings:
        imbalance_level = _max_level(imbalance_level, finding["level"])
        add(
            "diagnostics",
            finding["level"],
            finding["title"],
            finding["message"],
            details=finding.get("details", {}),
        )
        recommend("imbalance_distribution", "watch", "train/validation split의 클래스 분포가 비슷한지 확인하고, 필요하면 stratified split 또는 데이터 추가를 고려하세요.")

    class_level = _add_class_insights(add, recommend, class_rows)
    confusion_level = _add_confusion_insights(add, recommend, matrix, labels)
    danger_level = _add_danger_insights(add, recommend, class_rows, matrix, labels)
    data_quality_level = _add_data_quality_insights(add, recommend, data_stats)
    trend_level = _add_trend_insights(add, recommend, history_rows, metrics)
    loss_level = _add_validation_loss_insights(
        add,
        recommend,
        val_loss=val_loss,
        val_cross_entropy=val_cross_entropy,
        val_macro_f1=val_macro_f1,
        mean_pred_confidence=mean_pred_confidence,
        mean_true_confidence=mean_true_confidence,
        history_rows=history_rows,
    )

    badge("Overfitting Risk", overfit_level, _badge_detail(overfit_level, "loss gap"))
    badge("Generalization", performance_level, _metric_detail("macro F1", val_macro_f1))
    badge("Class Imbalance", imbalance_level, _badge_detail(imbalance_level, "distribution"))
    badge("Danger Recall", danger_level, _badge_detail(danger_level, "risk classes"))
    badge("Validation Loss", loss_level, _metric_detail("val loss", val_loss))
    badge("Data Quality", data_quality_level, _badge_detail(data_quality_level, "skip/pose"))
    if trend_level != "good":
        badge("Validation Trend", trend_level, _badge_detail(trend_level, "history"))
    elif history_rows:
        badge("Validation Trend", "good", "stable")

    overall_level = "good"
    for bucket in (
        "risk_badges",
        "diagnostics",
        "class_insights",
        "confusion_insights",
        "data_quality_insights",
        "trend_insights",
    ):
        for item in result.get(bucket, []):
            overall_level = _max_level(overall_level, item.get("level"))
    result["overall_level"] = overall_level
    result["overall_tone"] = _level_tone(overall_level)

    _build_summary(result, val_accuracy=val_accuracy, val_macro_f1=val_macro_f1, train_loss=train_loss, val_loss=val_loss)
    if not result["recommendations"] and overall_level == "good":
        recommend("good_continue", "good", "현재 지표가 안정적입니다. 다음 실험에서는 데이터 품질과 위험 클래스 recall을 계속 모니터링하세요.")

    return _trim_result(result)


def _add_metric_notes(
    add,
    *,
    val_accuracy: float | None,
    val_macro_f1: float | None,
    train_loss: float | None,
    val_loss: float | None,
    skipped_samples: int | None,
) -> None:
    if val_accuracy is not None:
        if val_accuracy < 0.5:
            message = f"전체 샘플 중 약 {val_accuracy * 100:.1f}%를 맞췄습니다. 클래스 불균형이 있으면 실제 체감 성능은 더 낮을 수 있습니다."
            level = "warning"
        else:
            message = f"전체 샘플 중 약 {val_accuracy * 100:.1f}%를 맞췄습니다. macro F1과 함께 봐야 클래스별 성능을 판단할 수 있습니다."
            level = "good"
        add("metric_notes", level, "Validation Accuracy", message, value=round(val_accuracy, 6))
    if val_macro_f1 is not None:
        if val_macro_f1 < 0.4:
            message = "클래스별 성능 평균이 낮은 편입니다. recall이 0에 가까운 클래스가 있는지 확인해야 합니다."
            level = "warning"
        else:
            message = "클래스별 평균 성능이 비교적 안정적입니다. 위험 클래스 recall은 별도로 확인하는 것이 좋습니다."
            level = "good"
        add("metric_notes", level, "Validation Macro F1", message, value=round(val_macro_f1, 6))
    if train_loss is not None and val_loss is not None:
        if train_loss <= 0.35 and val_loss >= 1.5:
            message = "학습 데이터에는 잘 맞지만 검증 데이터에는 일반화가 잘 되지 않는 과적합 가능성이 있습니다."
            level = "warning"
        else:
            message = "train/validation loss 격차를 epoch 추세와 함께 확인하세요."
            level = "good"
        add(
            "metric_notes",
            level,
            "Train / Validation Loss",
            message,
            details={"train_loss": train_loss, "val_loss": val_loss},
        )
    if skipped_samples is not None and skipped_samples > 0:
        level = "warning" if skipped_samples >= 100 else "watch"
        add(
            "metric_notes",
            level,
            "Skipped Samples",
            "스킵 샘플이 많으면 데이터 로딩, pose 품질 기준, 특정 클래스 편중 여부를 확인해야 합니다.",
            value=skipped_samples,
        )


def _add_class_insights(add, recommend, class_rows: list[dict]) -> str:
    if not class_rows:
        return "good"

    level = "good"
    supported = [row for row in class_rows if _safe_int(row.get("support")) not in (None, 0)]
    recall_rows = [
        row for row in supported
        if _safe_float(row.get("recall")) is not None
    ]
    low_recall = sorted(
        [row for row in recall_rows if (_safe_float(row.get("recall")) or 0.0) < 0.3],
        key=lambda row: (_safe_float(row.get("recall")) or 0.0, -(_safe_int(row.get("support")) or 0)),
    )
    if low_recall:
        worst = low_recall[0]
        worst_recall = _safe_float(worst.get("recall")) or 0.0
        level = "critical" if worst_recall < 0.1 else "warning"
        add(
            "class_insights",
            level,
            "낮은 클래스 recall",
            f"{worst.get('label')} 클래스 recall이 {worst_recall:.3f}로 낮아 실제 샘플을 놓칠 가능성이 큽니다.",
            details={"label": worst.get("label"), "recall": worst_recall, "support": worst.get("support")},
        )
        for row in low_recall[1:4]:
            recall = _safe_float(row.get("recall")) or 0.0
            add(
                "class_insights",
                "warning" if recall >= 0.1 else "critical",
                "추가 low recall 클래스",
                f"{row.get('label')} 클래스도 recall {recall:.3f}로 낮습니다.",
                details={"label": row.get("label"), "recall": recall, "support": row.get("support")},
            )
        recommend("low_recall_samples", "warning", "recall이 낮은 클래스의 false negative 샘플을 저장해 라벨 품질과 동작 패턴 차이를 확인하세요.")

    supports = [int(row.get("support") or 0) for row in supported]
    if supports:
        min_support = min(supports)
        max_support = max(supports)
        if min_support > 0 and max_support / max(min_support, 1) >= 5.0:
            level = _max_level(level, "warning")
            add(
                "class_insights",
                "warning",
                "클래스 support 불균형",
                f"validation support가 최대 {max_support} / 최소 {min_support}로 차이가 커 macro F1 변동이 커질 수 있습니다.",
                details={"min_support": min_support, "max_support": max_support},
            )
            recommend("support_balance", "watch", "클래스별 validation support가 너무 작지 않도록 split 또는 데이터 수집 기준을 점검하세요.")

    near_zero_f1 = [
        row for row in supported
        if (_safe_float(row.get("f1")) is not None and (_safe_float(row.get("f1")) or 0.0) < 0.1)
    ]
    if near_zero_f1:
        level = _max_level(level, "critical")
        labels = ", ".join(str(row.get("label")) for row in near_zero_f1[:4])
        add(
            "class_insights",
            "critical",
            "거의 학습되지 않은 클래스",
            f"{labels} 클래스의 F1이 0에 가까워 라벨 매핑, 데이터 수, augmentation을 우선 확인해야 합니다.",
            details={"labels": [row.get("label") for row in near_zero_f1]},
        )
    return level


def _add_danger_insights(add, recommend, class_rows: list[dict], matrix: list[list[int]], labels: list[str]) -> str:
    danger_indices = _danger_indices(labels, class_rows)
    if not danger_indices:
        return "good"

    level = "good"
    row_by_index = {
        int(row.get("class_index", index) or index): row
        for index, row in enumerate(class_rows)
        if isinstance(row, dict)
    }
    for index in danger_indices:
        row = row_by_index.get(index)
        recall = _safe_float(row.get("recall")) if isinstance(row, dict) else None
        label = labels[index] if 0 <= index < len(labels) else (row.get("label") if row else str(index))
        if recall is not None and recall < 0.5:
            item_level = "critical" if recall < 0.3 else "warning"
            level = _max_level(level, item_level)
            add(
                "class_insights",
                item_level,
                "위험 클래스 recall 부족",
                f"{label} 클래스 recall이 {recall:.3f}로 낮아 실제 위험 상황을 놓칠 수 있습니다.",
                details={"label": label, "recall": recall},
            )
            recommend("danger_recall_data", "critical", "danger/warning 계열 클래스의 데이터 수와 라벨 품질을 보강하고 false negative 사례를 따로 확인하세요.")
            recommend("danger_threshold", "watch", "운영 목적이 위험 감지라면 class threshold 또는 cost-sensitive loss를 별도 실험하세요.")

    normal_index = _normal_index(labels)
    if normal_index is not None and matrix:
        for danger_index in danger_indices:
            if danger_index >= len(matrix):
                continue
            row = matrix[danger_index]
            support = sum(row)
            normal_misses = row[normal_index] if normal_index < len(row) else 0
            if support > 0 and normal_misses / support >= 0.25:
                item_level = "critical" if normal_misses / support >= 0.5 else "warning"
                level = _max_level(level, item_level)
                danger_label = labels[danger_index] if danger_index < len(labels) else str(danger_index)
                normal_label = labels[normal_index] if normal_index < len(labels) else str(normal_index)
                add(
                    "confusion_insights",
                    item_level,
                    "위험 클래스를 normal로 오분류",
                    f"{danger_label} 클래스가 {normal_label}로 {normal_misses}건 오분류되어 위험 상황 누락 가능성이 있습니다.",
                    details={"from": danger_label, "to": normal_label, "count": normal_misses, "support": support},
                )
    return level


def _add_confusion_insights(add, recommend, matrix: list[list[int]], labels: list[str]) -> str:
    if not matrix:
        return "good"

    total = sum(sum(row) for row in matrix)
    if total <= 0:
        return "good"

    level = "good"
    diag = [row[index] if index < len(row) else 0 for index, row in enumerate(matrix)]
    supports = [sum(row) for row in matrix]
    predicted = Counter()
    for row in matrix:
        for index, value in enumerate(row):
            predicted[index] += value

    best_index = max(range(len(diag)), key=lambda index: diag[index], default=None)
    if best_index is not None and diag[best_index] > 0:
        add(
            "confusion_insights",
            "good",
            "가장 많이 맞춘 클래스",
            f"{_label_at(labels, best_index)} 클래스가 {diag[best_index]}건으로 가장 많이 맞았습니다.",
            details={"label": _label_at(labels, best_index), "correct": diag[best_index]},
        )

    worst_index = max(
        range(len(supports)),
        key=lambda index: max(supports[index] - (diag[index] if index < len(diag) else 0), 0),
        default=None,
    )
    if worst_index is not None:
        wrong = max(supports[worst_index] - (diag[worst_index] if worst_index < len(diag) else 0), 0)
        wrong_ratio = wrong / max(supports[worst_index], 1)
        if wrong > 0 and (wrong_ratio >= 0.2 or wrong >= 10):
            level = _max_level(level, "watch")
            add(
                "confusion_insights",
                "watch",
                "가장 많이 틀린 클래스",
                f"{_label_at(labels, worst_index)} 클래스에서 {wrong}건의 오분류가 발생했습니다.",
                details={
                    "label": _label_at(labels, worst_index),
                    "wrong": wrong,
                    "support": supports[worst_index],
                    "wrong_ratio": round(wrong_ratio, 6),
                },
            )

    top_pair = _top_confusion_pair(matrix)
    if top_pair and top_pair[2] / max(supports[top_pair[0]], 1) >= 0.15:
        source_index, target_index, count = top_pair
        pair_level = "warning" if count / max(supports[source_index], 1) >= 0.25 else "watch"
        level = _max_level(level, pair_level)
        add(
            "confusion_insights",
            pair_level,
            "자주 혼동되는 클래스 쌍",
            f"{_label_at(labels, source_index)} 클래스가 {_label_at(labels, target_index)}로 {count}건 혼동되었습니다.",
            details={"from": _label_at(labels, source_index), "to": _label_at(labels, target_index), "count": count},
        )
        recommend("top_confusion_pair", "watch", "가장 자주 혼동되는 클래스 쌍의 샘플을 나란히 비교해 라벨 기준과 특징 차이를 정리하세요.")

    normal_index = _normal_index(labels)
    if normal_index is not None:
        predicted_normal = int(predicted.get(normal_index, 0))
        true_normal = supports[normal_index] if normal_index < len(supports) else 0
        predicted_ratio = predicted_normal / max(total, 1)
        true_ratio = true_normal / max(total, 1)
        if predicted_ratio >= 0.5 and predicted_ratio - true_ratio >= 0.2:
            level = _max_level(level, "warning")
            add(
                "confusion_insights",
                "warning",
                "normal 과다 예측",
                "모델이 normal 클래스를 실제 분포보다 과도하게 예측하는 경향이 있습니다.",
                details={"predicted_ratio": round(predicted_ratio, 4), "true_ratio": round(true_ratio, 4)},
            )
            recommend("normal_bias", "warning", "normal로 예측된 false negative를 우선 확인하고 class weight 또는 threshold 조정을 실험하세요.")

    missing_predictions = [
        _label_at(labels, index)
        for index, support in enumerate(supports)
        if support > 0 and int(predicted.get(index, 0)) == 0
    ]
    if missing_predictions:
        level = _max_level(level, "critical")
        add(
            "confusion_insights",
            "critical",
            "거의 예측되지 않는 클래스",
            f"{', '.join(missing_predictions[:4])} 클래스가 validation에서 거의 예측되지 않았습니다.",
            details={"labels": missing_predictions},
        )
        recommend(
            "missing_predictions",
            "critical",
            "거의 예측되지 않는 클래스의 train 샘플 수, label mapping, augmentation 적용 여부를 먼저 확인하고 "
            "이전 checkpoint를 그대로 이어받기보다 focal loss, sampler, false negative 샘플 검토를 함께 실험하세요.",
        )

    return level


def _add_data_quality_insights(add, recommend, data_stats: dict) -> str:
    if not data_stats:
        return "good"

    level = "good"
    skip_reports = [
        ("현재 작업", data_stats.get("skip_report")),
        ("누적", data_stats.get("cumulative_skip_report")),
    ]
    for label, report in skip_reports:
        if not isinstance(report, dict):
            continue
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        prepare_summary = report.get("prepare_summary") if isinstance(report.get("prepare_summary"), dict) else {}
        ratio = _safe_float(summary.get("skip_ratio"))
        if ratio is None:
            overall = prepare_summary.get("overall") if isinstance(prepare_summary.get("overall"), dict) else {}
            ratio = _safe_float(overall.get("skip_ratio"))
        skipped = _safe_int(summary.get("skipped_items")) or _safe_int(summary.get("skipped_count")) or 0
        total = _safe_int(summary.get("total_items"))
        if ratio is None and total:
            ratio = skipped / max(total, 1)
        if ratio is not None and ratio >= 0.1:
            item_level = "critical" if ratio >= 0.25 else "warning"
            level = _max_level(level, item_level)
            add(
                "data_quality_insights",
                item_level,
                f"{label} 스킵 비율",
                f"{label} 데이터의 스킵 비율이 {ratio * 100:.1f}%로 높아 학습 분포가 왜곡될 수 있습니다.",
                details={"skip_ratio": round(ratio, 6), "skipped": skipped, "total": total},
            )
            recommend("skip_ratio", "warning", "skip reason 통계를 확인하고 min_frames, missing frame, fallback/padding 기준이 특정 클래스에 불리하지 않은지 점검하세요.")

        by_label = summary.get("by_label") or {}
        prepare_by_label = prepare_summary.get("by_label") if isinstance(prepare_summary.get("by_label"), dict) else {}
        worst_label, worst_payload = _worst_skip_label(prepare_by_label, by_label)
        worst_ratio = _safe_float(worst_payload.get("skip_ratio")) if isinstance(worst_payload, dict) else None
        if worst_label and worst_ratio is not None and worst_ratio >= 0.2:
            item_level = "critical" if worst_ratio >= 0.4 else "warning"
            level = _max_level(level, item_level)
            add(
                "data_quality_insights",
                item_level,
                "특정 클래스 스킵 편중",
                f"{worst_label} 클래스 스킵 비율이 {worst_ratio * 100:.1f}%로 높아 해당 클래스 학습량이 줄어들 수 있습니다.",
                details={"label": worst_label, "skip_ratio": round(worst_ratio, 6)},
            )
            recommend("class_skip_bias", "warning", "스킵이 몰린 클래스의 원본 영상 품질, pose confidence, 라벨/파일 매칭을 먼저 확인하세요.")

        by_reason = summary.get("by_reason") or {}
        if isinstance(by_reason, dict) and by_reason:
            reason, count = max(by_reason.items(), key=lambda item: int(item[1] or 0))
            if int(count or 0) > 0:
                add(
                    "data_quality_insights",
                    "watch",
                    "주요 스킵 원인",
                    f"가장 많은 스킵 원인은 {reason}이며 {int(count or 0)}건 발생했습니다.",
                    details={"reason": reason, "count": int(count or 0)},
                )

        recovery_actions = prepare_summary.get("recovery_actions") if isinstance(prepare_summary, dict) else {}
        if isinstance(recovery_actions, dict) and recovery_actions:
            recovered = sum(int(value or 0) for value in recovery_actions.values())
            if recovered > 0:
                add(
                    "data_quality_insights",
                    "watch",
                    "fallback/padding 사용",
                    f"pose 복구 또는 padding이 {recovered}건 사용되었습니다. 비율이 높으면 실제 동작 정보가 부족할 수 있습니다.",
                    details={"recovered": recovered, "actions": recovery_actions},
                )
                recommend("fallback_padding", "watch", "fallback/padding 샘플의 비율과 예측 실패 사례를 함께 확인하세요.")

    return level


def _add_trend_insights(add, recommend, history_rows: list[dict], metrics: dict) -> str:
    if len(history_rows) < 3:
        return "good"

    level = "good"
    train_loss_series = _series(history_rows, "train_loss")
    val_loss_series = _series(history_rows, "val_loss")
    val_f1_series = _series(history_rows, "val_macro_f1", "macro_f1")
    val_accuracy_series = _series(history_rows, "val_accuracy", "accuracy")

    if len(train_loss_series) >= 3 and len(val_loss_series) >= 3:
        train_delta = train_loss_series[-1] - train_loss_series[0]
        val_delta = val_loss_series[-1] - val_loss_series[0]
        if train_delta < -0.05 and val_delta > 0.05:
            level = _max_level(level, "warning")
            add(
                "trend_insights",
                "warning",
                "loss 추세 과적합",
                "train loss는 감소했지만 validation loss는 증가해 학습 후반 과적합이 진행된 것으로 보입니다.",
                details={"train_loss_delta": round(train_delta, 6), "val_loss_delta": round(val_delta, 6)},
            )
            recommend("trend_overfit", "warning", "best epoch 이후 checkpoint를 사용하고, regularization 또는 augmentation을 강화하세요.")

    if len(val_f1_series) >= 3:
        recent_delta = val_f1_series[-1] - val_f1_series[max(0, len(val_f1_series) - 4)]
        if recent_delta >= 0.02:
            add(
                "trend_insights",
                "good",
                "macro F1 개선 중",
                "최근 epoch에서 validation macro F1이 개선되고 있어 추가 학습 여지가 있습니다.",
                details={"recent_macro_f1_delta": round(recent_delta, 6)},
            )
        elif recent_delta <= -0.03:
            level = _max_level(level, "warning")
            add(
                "trend_insights",
                "warning",
                "macro F1 악화",
                "최근 epoch에서 validation macro F1이 떨어져 best epoch 이후 성능 악화 가능성이 있습니다.",
                details={"recent_macro_f1_delta": round(recent_delta, 6)},
            )

    if len(val_accuracy_series) >= 3 and len(val_f1_series) >= 3:
        acc_delta = val_accuracy_series[-1] - val_accuracy_series[0]
        f1_delta = val_f1_series[-1] - val_f1_series[0]
        if acc_delta >= 0.03 and abs(f1_delta) < 0.015:
            level = _max_level(level, "watch")
            add(
                "trend_insights",
                "watch",
                "accuracy만 개선",
                "accuracy는 개선되었지만 macro F1은 정체되어 특정 클래스 문제는 아직 남아 있을 수 있습니다.",
                details={"accuracy_delta": round(acc_delta, 6), "macro_f1_delta": round(f1_delta, 6)},
            )

    best_epoch = _safe_int(metrics.get("best_epoch"))
    last_epoch = _safe_int(history_rows[-1].get("epoch"))
    best_f1 = _safe_float(metrics.get("best_val_macro_f1"))
    last_f1 = val_f1_series[-1] if val_f1_series else None
    if best_epoch is not None and last_epoch is not None and best_epoch < last_epoch and best_f1 is not None and last_f1 is not None:
        if best_f1 - last_f1 >= 0.03:
            level = _max_level(level, "watch")
            add(
                "trend_insights",
                "watch",
                "best epoch 이후 성능 하락",
                "best epoch 이후 validation macro F1이 낮아져 early stopping 또는 best checkpoint 사용이 적절합니다.",
                details={"best_epoch": best_epoch, "last_epoch": last_epoch, "best_macro_f1": best_f1, "last_macro_f1": last_f1},
            )

    return level


def _add_validation_loss_insights(
    add,
    recommend,
    *,
    val_loss: float | None,
    val_cross_entropy: float | None,
    val_macro_f1: float | None,
    mean_pred_confidence: float | None,
    mean_true_confidence: float | None,
    history_rows: list[dict],
) -> str:
    effective_loss = val_cross_entropy if val_cross_entropy is not None else val_loss
    if effective_loss is None:
        return "good"

    level = "good"
    if effective_loss >= 1.5:
        item_level = "critical" if effective_loss >= 2.5 else "warning"
        level = _max_level(level, item_level)
        add(
            "diagnostics",
            item_level,
            "validation loss 높음",
            "validation loss가 높아 일반화 문제, noisy label, 확률 보정 문제를 확인해야 합니다.",
            details={"validation_loss": effective_loss},
        )
        recommend("high_val_loss", "warning", "validation transform/label mapping을 다시 확인하고 label smoothing 또는 calibration을 검토하세요.")

    if effective_loss >= 1.5 and mean_pred_confidence is not None and mean_pred_confidence >= 0.8:
        level = _max_level(level, "warning")
        add(
            "diagnostics",
            "warning",
            "높은 confidence와 높은 loss",
            "모델이 틀린 예측을 과도하게 확신하고 있을 수 있어 calibration 확인이 필요합니다.",
            details={"validation_loss": effective_loss, "mean_pred_confidence": mean_pred_confidence},
        )
        recommend("confidence_calibration", "watch", "temperature scaling, label smoothing, noisy label 점검으로 confidence calibration을 확인하세요.")

    if effective_loss >= 1.5 and mean_true_confidence is not None and mean_true_confidence < 0.45:
        level = _max_level(level, "watch")
        add(
            "diagnostics",
            "watch",
            "정답 클래스 confidence 낮음",
            "정답 클래스 평균 confidence가 낮아 클래스 경계가 불안정하거나 라벨이 섞였을 수 있습니다.",
            details={"mean_true_confidence": mean_true_confidence},
        )

    val_f1_series = _series(history_rows, "val_macro_f1", "macro_f1")
    val_loss_series = _series(history_rows, "val_loss")
    if len(val_f1_series) >= 3 and len(val_loss_series) >= 3:
        f1_delta = val_f1_series[-1] - val_f1_series[max(0, len(val_f1_series) - 4)]
        loss_delta = val_loss_series[-1] - val_loss_series[max(0, len(val_loss_series) - 4)]
        if effective_loss >= 1.5 and f1_delta >= 0.02:
            add(
                "trend_insights",
                "watch",
                "loss는 높지만 F1 개선",
                "validation loss는 높지만 macro F1이 개선되어 확률 보정 문제일 가능성이 있습니다.",
                details={"recent_macro_f1_delta": round(f1_delta, 6), "recent_loss_delta": round(loss_delta, 6)},
            )
            recommend("loss_f1_calibration", "watch", "macro F1과 loss가 엇갈리면 confidence calibration과 threshold를 별도로 확인하세요.")
        if loss_delta > 0.03 and f1_delta < -0.02:
            level = _max_level(level, "warning")
            add(
                "trend_insights",
                "warning",
                "loss와 F1 동시 악화",
                "validation loss는 증가하고 macro F1은 감소해 일반화 성능 저하 가능성이 있습니다.",
                details={"recent_macro_f1_delta": round(f1_delta, 6), "recent_loss_delta": round(loss_delta, 6)},
            )
    return level


def _build_summary(result: dict, *, val_accuracy: float | None, val_macro_f1: float | None, train_loss: float | None, val_loss: float | None) -> None:
    summary: list[dict] = []
    overall_level = result.get("overall_level") or "good"
    if overall_level == "critical":
        summary.append(
            _summary_item(
                "critical",
                "진단 요약",
                "현재 학습 결과에는 즉시 확인해야 할 위험 신호가 있습니다. 낮은 macro F1, 특정 클래스 recall, validation loss를 우선 점검하세요.",
            )
        )
    elif overall_level in {"warning", "watch"}:
        summary.append(
            _summary_item(
                overall_level,
                "진단 요약",
                "현재 모델은 일부 지표에서 주의가 필요합니다. accuracy만 보지 말고 macro F1, class recall, confusion matrix를 함께 확인하세요.",
            )
        )
    else:
        summary.append(
            _summary_item(
                "good",
                "진단 요약",
                "현재 핵심 지표는 비교적 안정적입니다. 다음 실험에서도 위험 클래스 recall과 데이터 품질을 계속 모니터링하세요.",
            )
        )

    if train_loss is not None and val_loss is not None and train_loss <= 0.35 and val_loss >= 1.5:
        summary.append(
            _summary_item(
                "warning",
                "과적합 가능성",
                "train loss에 비해 validation loss가 높아 regularization, augmentation, early stopping 확인이 필요합니다.",
            )
        )
    if val_accuracy is not None and val_macro_f1 is not None and val_accuracy - val_macro_f1 >= 0.2:
        summary.append(
            _summary_item(
                "warning",
                "클래스 편향 가능성",
                "accuracy보다 macro F1이 낮아 다수 클래스 위주 예측 또는 minority class 성능 저하가 의심됩니다.",
            )
        )

    priority_sources = (
        result.get("class_insights") or [],
        result.get("confusion_insights") or [],
        result.get("data_quality_insights") or [],
        result.get("trend_insights") or [],
    )
    for bucket in priority_sources:
        for item in bucket:
            if item.get("level") in {"critical", "warning"}:
                summary.append(_summary_item(item.get("level"), item.get("title"), item.get("message")))
                break
        if len(summary) >= 5:
            break

    result["summary"] = summary[:5]


def _summary_item(level: str, title: str, message: str) -> dict:
    return {
        "level": _normalize_level(level),
        "tone": _level_tone(level),
        "title": title,
        "message": message,
    }


def _minority_signal_recommendation(metrics: dict) -> str:
    class_weight_mode = str(metrics.get("class_weight_mode") or "").strip().lower()
    sampler_mode = str(metrics.get("balanced_sampler") or "").strip().lower()
    loss_name = str(metrics.get("loss_name") or "").strip().lower()
    off_values = {"", "off", "none", "false", "disabled", "0", "auto_off"}
    active = []
    if class_weight_mode not in off_values:
        active.append("class weight")
    if "multiplier" in class_weight_mode:
        active.append("cost-sensitive class multipliers")
    if sampler_mode not in off_values and not sampler_mode.endswith("_unavailable"):
        active.append("WeightedRandomSampler")
    if loss_name in {"focal", "focal_loss"}:
        active.append("focal loss")

    if not active:
        return "class weight, WeightedRandomSampler, focal loss 중 하나를 실험해 minority class 학습 신호를 키우세요."
    if loss_name not in {"focal", "focal_loss"}:
        return (
            f"{', '.join(active)}는 이미 적용 중입니다. "
            "다음 비교 실험은 loss를 focal로 바꿔 minority class의 어려운 샘플 학습 신호가 커지는지 확인하세요."
        )
    return (
        f"{', '.join(active)}가 이미 적용 중입니다. "
        "추가 개선은 부족 클래스 데이터 추가, 라벨 품질 점검, false negative 샘플 검토, class threshold/cost-sensitive 설정 비교를 우선하세요."
    )


def _distribution_findings(data_stats: dict) -> list[dict]:
    findings = []
    for key, title in (
        ("train_distribution", "학습 데이터 분포"),
        ("val_distribution", "검증 데이터 분포"),
    ):
        payload = data_stats.get(key)
        if not isinstance(payload, dict):
            continue
        severity = str(payload.get("severity") or "").strip().lower()
        if severity in {"critical", "warning"}:
            level = "critical" if severity == "critical" else "warning"
            messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
            findings.append(
                {
                    "level": level,
                    "title": title,
                    "message": " ".join(str(message) for message in messages[:2]) or "클래스 분포가 불균형합니다.",
                    "details": {
                        "severity": severity,
                        "covered": payload.get("covered"),
                        "total": payload.get("total"),
                        "empty_labels": payload.get("empty_labels") or [],
                        "low_sample_labels": payload.get("low_sample_labels") or [],
                        "imbalance_ratio": payload.get("imbalance_ratio"),
                        "dominant_label": payload.get("dominant_label"),
                        "dominant_count": payload.get("dominant_count"),
                        "minority_label": payload.get("minority_label"),
                        "minority_count": payload.get("minority_count"),
                    },
                }
            )
    return findings


def _normalize_history(history: Any) -> list[dict]:
    if not isinstance(history, list):
        return []
    return [row for row in history if isinstance(row, dict)]


def _normalize_labels(labels: Any) -> list[str]:
    if not isinstance(labels, list):
        return []
    return [str(label) for label in labels if str(label or "").strip()]


def _labels_from_class_report(class_report: Any) -> list[str]:
    labels = []
    if not isinstance(class_report, list):
        return labels
    for row in class_report:
        if isinstance(row, dict) and row.get("label") is not None:
            labels.append(str(row.get("label")))
    return labels


def _normalize_class_report(rows: Any, *, labels: list[str]) -> list[dict]:
    if not isinstance(rows, list):
        return []
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        class_index = _safe_int(row.get("class_index"))
        if class_index is None:
            class_index = index
        label = row.get("label")
        if label is None and 0 <= class_index < len(labels):
            label = labels[class_index]
        normalized.append(
            {
                **row,
                "class_index": class_index,
                "label": str(label if label is not None else class_index),
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1")),
                "support": _safe_int(row.get("support")),
            }
        )
    return normalized


def _normalize_matrix(matrix: Any) -> list[list[int]]:
    if not isinstance(matrix, list):
        return []
    normalized = []
    for row in matrix:
        if not isinstance(row, list):
            continue
        normalized.append([max(_safe_int(value) or 0, 0) for value in row])
    return normalized


def _metric(*containers: Any, keys: tuple[str, ...]) -> float | None:
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = _safe_float(container.get(key))
            if value is not None:
                return value
    return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number


def _series(rows: list[dict], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in keys:
            value = _safe_float(row.get(key))
            if value is not None:
                values.append(value)
                break
    return values


def _danger_indices(labels: list[str], class_rows: list[dict]) -> list[int]:
    indices: set[int] = set()
    for index, label in enumerate(labels):
        if _is_danger_label(label):
            indices.add(index)
    for row in class_rows:
        label = str(row.get("label") or "")
        class_index = _safe_int(row.get("class_index"))
        if class_index is not None and _is_danger_label(label):
            indices.add(class_index)
    return sorted(indices)


def _is_danger_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    return any(keyword in normalized for keyword in DANGER_LABEL_KEYWORDS)


def _normal_index(labels: list[str]) -> int | None:
    for index, label in enumerate(labels):
        normalized = str(label or "").strip().lower()
        if any(keyword in normalized for keyword in NORMAL_LABEL_KEYWORDS):
            return index
    return None


def _top_confusion_pair(matrix: list[list[int]]) -> tuple[int, int, int] | None:
    best: tuple[int, int, int] | None = None
    for source_index, row in enumerate(matrix):
        for target_index, count in enumerate(row):
            if source_index == target_index or count <= 0:
                continue
            if best is None or count > best[2]:
                best = (source_index, target_index, count)
    return best


def _label_at(labels: list[str], index: int) -> str:
    return labels[index] if 0 <= index < len(labels) else str(index)


def _worst_skip_label(prepare_by_label: Any, by_label_counts: Any) -> tuple[str | None, dict]:
    if isinstance(prepare_by_label, dict) and prepare_by_label:
        label, payload = max(
            prepare_by_label.items(),
            key=lambda item: (_safe_float(item[1].get("skip_ratio")) or 0.0, _safe_int(item[1].get("skipped")) or 0)
            if isinstance(item[1], dict)
            else (0.0, 0),
        )
        return str(label), payload if isinstance(payload, dict) else {}
    if isinstance(by_label_counts, dict) and by_label_counts:
        label, count = max(by_label_counts.items(), key=lambda item: int(item[1] or 0))
        return str(label), {"skipped": int(count or 0), "skip_ratio": None}
    return None, {}


def _skip_total(data_stats: dict) -> int | None:
    report = data_stats.get("skip_report") if isinstance(data_stats, dict) else None
    if not isinstance(report, dict):
        return None
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return _safe_int(summary.get("skipped_items")) or _safe_int(summary.get("skipped_count")) or _safe_int(summary.get("total_issues"))


def _normalize_level(level: Any) -> str:
    normalized = str(level or "good").strip().lower()
    if normalized == "ok":
        return "good"
    if normalized not in LEVEL_ORDER:
        return "watch"
    return normalized


def _level_tone(level: Any) -> str:
    return LEVEL_TONES.get(_normalize_level(level), "warn")


def _max_level(left: Any, right: Any) -> str:
    left_level = _normalize_level(left)
    right_level = _normalize_level(right)
    return right_level if LEVEL_ORDER[right_level] > LEVEL_ORDER[left_level] else left_level


def _metric_detail(label: str, value: float | None) -> str:
    return f"{label} {value:.3f}" if value is not None else ""


def _badge_detail(level: str, fallback: str) -> str:
    labels = {
        "good": "Good",
        "normal": "Good",
        "improving": "Improving",
        "watch": "Watch",
        "warning": "Warning",
        "critical": "Critical",
    }
    return labels.get(_normalize_level(level), fallback)


def _trim_result(result: dict) -> dict:
    limits = {
        "summary": 5,
        "risk_badges": 8,
        "diagnostics": 8,
        "metric_notes": 6,
        "class_insights": 8,
        "confusion_insights": 8,
        "data_quality_insights": 8,
        "trend_insights": 8,
        "recommendations": 8,
    }
    for key, limit in limits.items():
        items = result.get(key)
        if isinstance(items, list):
            result[key] = items[:limit]
    return result
