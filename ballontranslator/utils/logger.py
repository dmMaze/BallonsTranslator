import datetime
import logging
import os
import os.path as osp
from glob import glob
import termcolor


if os.name == "nt":  # Windows
    import colorama
    colorama.init()


COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, use_color=True):
        logging.Formatter.__init__(self, fmt)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:

            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored("{:<7}".format(record.levelname))
            record.message2 = colored(record.getMessage())

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        return logging.Formatter.format(self, record)

FORMAT = (
    "[%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"
)

class ColoredLogger(logging.Logger):

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.WARNING)

        color_formatter = ColoredFormatter(FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return


def setup_logging(logfile_dir: str, max_num_logs=14):

    if not osp.exists(logfile_dir):
        os.makedirs(logfile_dir)
    else:
        old_logs = glob(osp.join(logfile_dir, '*.log'))
        old_logs.sort()
        n_log = len(old_logs)
        if n_log >= max_num_logs:
            to_remove = n_log - max_num_logs + 1
            try:
                for ii in range(to_remove):
                    os.remove(old_logs[ii])
            except Exception as e:
                logger.error(e)

    logfilename = datetime.datetime.now().strftime('_%Y_%m_%d-%H_%M_%S.log')
    logfilep = osp.join(logfile_dir, logfilename)
    fh = logging.FileHandler(logfilep, mode='w', encoding='utf-8')
    fh.setFormatter(
        logging.Formatter(
            ("[%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s")
        )
    )
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger('BallonTranslator')
logger.setLevel(logging.DEBUG)
logger.propagate = False
