from fedrda_experiments.config import parse_args
from fedrda_experiments.runner import ExperimentRunner


if __name__ == "__main__":
    configuration, resume = parse_args()
    directory = ExperimentRunner(configuration, resume=resume).run()
    print(f"completed: {directory}")
