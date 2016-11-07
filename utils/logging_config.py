import logging

import settings


def logging_config(level=settings.LOGGING_LEVEL, format='%(asctime)s [%(process)d] %(levelname)s %(message)s',
                   filename=settings.LOGGING_FILENAME, filemode='a'):
    root_logger = logging.getLogger()
    map(root_logger.removeHandler, root_logger.handlers[:])
    map(root_logger.removeFilter, root_logger.filters[:])
    if filename:
        logging.basicConfig(level=level, format=format, filename=filename, filemode=filemode)
    else:
        logging.basicConfig(level=level, format=format)
