from .tools import mkdir_if_missing
import sys
import os.path as osp
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
from args import argument_parser
parser = argument_parser()
args = parser.parse_args()
class Logger:
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d--%H%M%S")
        log_dir = "/content/drive/MyDrive/TensorBoard_Logs/" + formatted_datetime + "_" + args.name
        self.console = sys.stdout
        self.file = None
        self.writer = SummaryWriter(log_dir=log_dir)
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def create_summary_writer(self):
        return self.writer
    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

