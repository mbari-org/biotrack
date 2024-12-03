# biotrack, CC-BY-NC license
# Filename: __main__.py
# Description: Main entry point for the biotrack command line interface
from datetime import datetime
from pathlib import Path

import click
import sys

from biotrack.logger import create_logger_file, info, err
from biotrack import __version__

sys.path.insert(0, str(Path(__file__).parent.parent)) 


create_logger_file("biotrack")
default_data_path = Path(__file__).parent / "testdata"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", message="%(prog)s, version %(version)s")
def cli():
    """
    Track video from a command line.
    """
    pass


if __name__ == "__main__":
    try:
        start = datetime.now()
        cli()
        end = datetime.now()
        info(f"Done. Elapsed time: {end - start} seconds")
    except Exception as e:
        err(f"Exiting. Error: {e}")
        exit(-1)
