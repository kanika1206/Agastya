from agastya.stages.base import Stage


class Doubler(Stage[int, int]):
    name = "doubler"

    def process(self, item: int) -> int:
        return item * 2


def test_stage_process_runs():
    assert Doubler().process(3) == 6


def test_stage_exposes_name():
    assert Doubler().name == "doubler"
