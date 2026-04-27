"""Task t07: Validate new model vs production"""

import json

from clearml import Task
from shared.clearml_utils import get_production_model
from shared.config import CLEARML_PROJECT_NAME, THRESHOLDS_PATH


def main():
    task = Task.init(
        project_name=CLEARML_PROJECT_NAME,
        task_name="t07_validate_model",
        task_type=Task.TaskTypes.data_processing,
    )
    logger = task.get_logger()

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)["validation"]

    dbcv_min_relative = thresholds["dbcv_min_relative"]
    cv_min_abs = thresholds["cv_coherence_absolute_min"]
    noise_max = thresholds["noise_ratio_max"]
    num_topics_min = thresholds["num_topics_min"]

    # Load new model training meta
    training_meta_path = task.artifacts["training_meta.json"].get_local_copy()
    with open(training_meta_path) as f:
        training_meta = json.load(f)

    new_metrics = training_meta["metrics"]
    dbcv_new = float(new_metrics.get("dbcv_score", 0))
    cv_new = float(new_metrics.get("cv_coherence", 0))
    noise_new = float(new_metrics.get("noise_ratio", 1))
    num_topics_new = int(new_metrics.get("num_topics", 0))

    # Get production metrics (if exists)
    prod_model_obj = get_production_model()
    cold_start = prod_model_obj is None
    dbcv_prod = None

    if not cold_start:
        try:
            prod_task = Task.get_task(task_id=prod_model_obj.task)
            prod_meta_path = prod_task.artifacts["training_meta.json"].get_local_copy()
            with open(prod_meta_path) as f:
                prod_meta = json.load(f)
            dbcv_prod = float(prod_meta["metrics"].get("dbcv_score", 0))
        except Exception as e:
            print(f"WARNING: could not load production metrics: {e}")
            cold_start = True

    checks: dict[str, dict] = {}

    # DBCV check
    if cold_start or dbcv_prod is None:
        dbcv_passed = True
        checks["dbcv"] = {
            "new": dbcv_new,
            "production": None,
            "passed": True,
            "note": "cold_start",
        }
    else:
        dbcv_passed = dbcv_new >= dbcv_prod * dbcv_min_relative
        checks["dbcv"] = {
            "new": dbcv_new,
            "production": dbcv_prod,
            "threshold_relative": dbcv_min_relative,
            "passed": dbcv_passed,
        }

    # C_V check (absolute)
    cv_passed = cv_new >= cv_min_abs
    checks["cv_coherence"] = {
        "new": cv_new,
        "threshold": cv_min_abs,
        "passed": cv_passed,
    }

    # Noise ratio check
    noise_passed = noise_new <= noise_max
    checks["noise_ratio"] = {
        "new": noise_new,
        "threshold": noise_max,
        "passed": noise_passed,
    }

    # Num topics check
    topics_passed = num_topics_new >= num_topics_min
    checks["num_topics"] = {
        "new": num_topics_new,
        "threshold": num_topics_min,
        "passed": topics_passed,
    }

    verdict = "passed" if all(c["passed"] for c in checks.values()) else "failed"

    # Log deltas
    if dbcv_prod is not None:
        logger.report_scalar(
            "validation", "dbcv_delta", value=dbcv_new - dbcv_prod, iteration=0
        )
    logger.report_scalar("validation", "cv_coherence_new", value=cv_new, iteration=0)
    logger.report_scalar("validation", "noise_ratio_new", value=noise_new, iteration=0)
    logger.report_scalar(
        "validation", "num_topics_new", value=num_topics_new, iteration=0
    )
    task.connect({"validation_verdict": verdict}, name="validation")

    validation_report = {"verdict": verdict, "checks": checks}
    report_path = "/tmp/validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    task.upload_artifact("validation_report.json", artifact_object=report_path)

    print(f"Validation verdict: {verdict}")
    for check_name, check_data in checks.items():
        status = "PASS" if check_data["passed"] else "FAIL"
        print(f"  [{status}] {check_name}: {check_data}")

    task.close()


if __name__ == "__main__":
    main()
