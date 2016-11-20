import logging

import settings


def logging_reconfig():
    logging.basicConfig(**settings.LOGGING_BASIC_CONFIG)
    logging.getLogger().addHandler(logging.StreamHandler())
    return None