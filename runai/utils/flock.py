import fcntl
import os

class Flock(object):
    """ A utility class for advisory lock files
    """

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)

        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except (IOError, OSError):
            os.close(self._fd)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)

# NOTE:
# `flock()` did not work over NFS in Linux kernels up to 2.6.11 but works in later Linux kernels.
# This is discussed around the internet and is documented here: https://www.man7.org/linux/man-pages/man2/flock.2.html.
# https://linux.die.net/man/2/flock is outdated and says "...flock() does not lock files over NFS. Use fcntl(2) instead...".
