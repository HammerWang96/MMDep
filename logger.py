import sys
import os
import logging


class Logger(object):
    def __init__(self, filename):
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%m-%d %H:%M')  # 指定日志格式
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()


if __name__ == '__main__':
    log = Logger('./data/tky/tky_log/NB.log')
    for i in range(5):
        log.info('{0},loss:{1}'.format(i, i+100))


