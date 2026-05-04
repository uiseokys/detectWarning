from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from dashboard_runtime import classify_job_exit


class DashboardRuntimeTests(unittest.TestCase):
    def test_data_ready_pipeline_status_is_not_marked_as_trained(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workspace = repo_root / ".tmp_test_dashboard_runtime_data_ready"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir()
        try:
            artifacts = workspace / "artifacts"
            artifacts.mkdir()
            pipeline_status = workspace / "pipeline_status.json"
            training_progress = workspace / "training_progress.json"
            pipeline_status.write_text(
                json.dumps(
                    {
                        "state": "data_ready",
                        "stage": "data_ready",
                        "message": "데이터 준비 완료, 학습 대기",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            training_progress.write_text("{}", encoding="utf-8")

            state, message = classify_job_exit(
                {
                    "pipeline_status": pipeline_status,
                    "training_progress": training_progress,
                    "artifacts_dir": artifacts,
                },
                {"filekey": "49658"},
                0,
            )

            self.assertEqual(state, "data_ready")
            self.assertIn("학습 대기", message)
        finally:
            if workspace.resolve().is_relative_to(repo_root.resolve()):
                shutil.rmtree(workspace, ignore_errors=True)

    def test_completed_artifacts_are_not_marked_as_warning_for_nonzero_exit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        workspace = repo_root / ".tmp_test_dashboard_runtime"
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir()
        try:
            artifacts = workspace / "artifacts"
            artifacts.mkdir()
            (artifacts / "best_action_model.pt").write_bytes(b"checkpoint")
            (artifacts / "metrics.json").write_text("{}", encoding="utf-8")
            (artifacts / "labels.json").write_text("{}", encoding="utf-8")
            pipeline_status = workspace / "pipeline_status.json"
            training_progress = workspace / "training_progress.json"
            log_path = workspace / "job.log"
            pipeline_status.write_text(json.dumps({"state": "completed", "stage": "completed"}), encoding="utf-8")
            training_progress.write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            log_path.write_text("[train] best model: best_action_model.pt\n", encoding="utf-8")

            state, _message = classify_job_exit(
                {
                    "pipeline_status": pipeline_status,
                    "training_progress": training_progress,
                    "artifacts_dir": artifacts,
                },
                {"log_path": str(log_path), "auto_recommended": True},
                1,
            )

            self.assertEqual(state, "completed")
        finally:
            if workspace.resolve().is_relative_to(repo_root.resolve()):
                shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
