import unittest

from ueler.export.job import Job, JobItem, JobState


class ExportJobTests(unittest.TestCase):
    def test_job_success_merges_metadata(self) -> None:
        calls = []

        def worker():
            calls.append("item1")
            return {"output_path": "/tmp/item1.png", "metadata": {"step": 1}}

        item = JobItem(
            item_id="item1",
            execute=worker,
            output_path="/tmp/placeholder.png",
            metadata={"mode": "full"},
        )
        job = Job(
            mode="full_fov",
            items=[item],
            marker_set="test",
            output_dir="/tmp",
            file_format="png",
        )

        status = job.start()
        result = status.results["item1"]

        self.assertEqual(calls, ["item1"])
        self.assertTrue(result.ok)
        self.assertEqual(result.output_path, "/tmp/item1.png")
        self.assertEqual(result.metadata["mode"], "full")
        self.assertEqual(result.metadata["step"], 1)
        self.assertEqual(status.state, JobState.COMPLETED)
        self.assertEqual(status.succeeded, 1)
        self.assertEqual(status.failed, 0)

    def test_job_failure_captures_traceback(self) -> None:
        def worker():
            raise RuntimeError("boom")

        item = JobItem(
            item_id="item1",
            execute=worker,
            output_path="/tmp/item1.png",
        )
        job = Job(
            mode="full_fov",
            items=[item],
            marker_set="test",
            output_dir="/tmp",
            file_format="png",
        )

        status = job.start()
        result = status.results["item1"]

        self.assertFalse(result.ok)
        self.assertIn("boom", result.error)
        self.assertTrue(result.traceback)
        self.assertEqual(status.failed, 1)
        self.assertEqual(status.succeeded, 0)
        self.assertEqual(status.state, JobState.COMPLETED)

    def test_job_cancel_halts_remaining_items(self) -> None:
        calls = []
        progress_events = []
        job_holder = {}

        def worker_one():
            calls.append("one")

        def worker_two():
            calls.append("two")

        items = [
            JobItem(item_id="first", execute=worker_one),
            JobItem(item_id="second", execute=worker_two),
        ]

        def progress_callback(status):
            progress_events.append(status.completed)
            if status.completed == 1:
                job_holder["job"].cancel()

        job = Job(
            mode="full_fov",
            items=items,
            marker_set="test",
            output_dir="/tmp",
            file_format="png",
            progress_callback=progress_callback,
        )
        job_holder["job"] = job

        status = job.start()

        self.assertEqual(calls, ["one"])
        self.assertEqual(status.state, JobState.CANCELLED)
        self.assertEqual(status.completed, 1)
        self.assertEqual(status.succeeded, 1)
        self.assertEqual(status.failed, 0)
        self.assertEqual(progress_events[0], 1)

    def test_status_before_start_is_pending(self) -> None:
        job = Job(
            mode="full_fov",
            items=[],
            marker_set="test",
            output_dir="/tmp",
            file_format="png",
        )
        status = job.status()
        self.assertEqual(status.state, JobState.PENDING)
        self.assertEqual(status.total, 0)
        self.assertEqual(status.completed, 0)
        self.assertEqual(status.succeeded, 0)
        self.assertEqual(status.failed, 0)


if __name__ == "__main__":
    unittest.main()
