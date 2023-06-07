LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'stdout': {
            'format': "%(asctime)s\t%(levelname)s\t%(name)s\t" \
                        "%(filename)s.%(funcName)s " \
                        "line: %(lineno)d | \t%(message)s",
        }
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'stdout',

        },
        "file":{
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'stdout',
            "filename": "log.log"
        }

    },
    'loggers': {
        'train': {
            'level': 'INFO',
            'handlers': ['stdout', "file"],
            'propogate': False
        }
    },
}