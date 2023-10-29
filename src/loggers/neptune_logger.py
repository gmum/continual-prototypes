import os

try:
    import neptune
except ImportError:
    print("Neptune is not installed. Cannot use neptune logger.")

import torch

from loggers.exp_logger import ExperimentLogger


def _convert_to_neptune_friendly_value(value):
    if torch.is_tensor(value):
        return value.item()

    if value is None:
        return "None"

    if isinstance(value, list):
        return str(value)

    if isinstance(value, dict):
        return {k: _convert_to_neptune_friendly_value(v) for k, v in value.items()}

    return value


class Logger(ExperimentLogger):
    """Characterizes a neptune logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)

        neptune_api_token = os.environ.get("NEPTUNE_API_TOKEN", "")
        neptune_account = os.environ.get("NEPTUNE_ACCOUNT", "")
        if not neptune_api_token or not neptune_account:
            raise ValueError(
                "NEPTUNE_API_TOKEN or NEPTUNE_ACCOUNT not set. "
                "If you want to use neptune.ai logger, set these env variables. "
                'Otherwise, change the "--log" script parameters so neptune is not used.'
            )
        self.neptune_run = neptune.init_run(
            project=f"{neptune_account}/continual-prototypes",
            api_token=neptune_api_token,
            name=exp_name,
        )
        print(f"Initiated neptune run")

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        if group is None:
            group = "none"

        self.neptune_run[f"{group}/{task}/{name}"].append(
            _convert_to_neptune_friendly_value(value)
        )

    def log_args(self, args):
        neptune_tags = [args.approach, args.network, args.datasets]

        if args.freeze_after_first_task:
            neptune_tags.append("freeze_after_first_task")
        if args.share_prototypes_between_tasks:
            neptune_tags.append("share_prototypes")

        for tag in neptune_tags:
            self.neptune_run["sys/tags"].add(tag)
        params = {
            k: _convert_to_neptune_friendly_value(v) for k, v in args.__dict__.items()
        }
        self.neptune_run["parameters"] = params

    def log_result(self, array, name, step):
        # Do not log results (numpy arrays) to neptune to save space
        pass

    def log_figure(self, name, iter, figure, curtime=None):
        self.neptune_run[f"figure/{name}/{iter}"].upload(figure)

    def save_model(self, state_dict, task):
        # do not save models in neptune to save space
        pass

    def __del__(self):
        self.neptune_run.stop()
