# configure global parameters, loggers, constants, etc
import logging.config

STR_PRECISION = None
REPR_PRECISION = None

LOGGING = {
    'version': 1,
    # 'disable_existing_loggers': True,
    'handlers': {
        'file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'warnings.log',
            'maxBytes': 65536,
        },
        'console': {
            'level': 'ERROR',
            'class': 'logging.StreamHandler',
        }
    },
    'loggers': {
        # default logger sends messages to a file
        '': {
            'level': 'INFO',
            'handlers': ['file'],
            'filename': 'info.log',
            'propagate': True,
        },
        # alternative logger is less verbose and sends messages to console
        'console': {
            'level': 'WARNING',
            'stream': 'ext://sys.stderr',
            'handlers': ['console'],
            'propagate': False,
        },
    },
}

logging.config.dictConfig(LOGGING)

#def logger_level(level='INFO'):
#    i = 0 
#    while i < logger_level.N  logger_level.order[i] != level:
#        i += 1
#    j = max(i - 1, 0)
#    for k in LOGGING['loggers']
#        LOGGING['loggers']['k']['level'] = logger_level.order[
#    logging.config.dictConfig(LOGGING)
#logger_level.order = (('DEBUG', logging.DEBUG), ('INFO', logging.DEBUG), ('WARNING', logging.WARNING), ('ERROR', logging.ERROR), ('CRITICAL', logging.ERROR)) ('NOTSET', logging.NOTSET)).
#logger_level.N = 6

