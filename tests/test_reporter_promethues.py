import os
import string
import unittest

from runai.utils import Hook, random
from runai.reporter import report_promethues

def random_args():
    reporter_name = random.string(chars=string.ascii_letters + string.digits)
    reporter_value = random.number()
    report_type = random.choice([_ for _ in report_promethues.ReportType])

    return (reporter_name, reporter_value, report_type)

def pid_exists(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

class Mock(Hook):
    def __init__(self, error=None):
        super(Mock, self).__init__(report_promethues, 'pushadd_to_gateway')
        self._error = error
        self.count = 0

    def __hook__(self, *args, **kwargs):
        self.count += 1
        if self._error:
            raise self._error()

class ReporterPromethuesBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["podUUID"] = random.string()
        os.environ["reporterGatewayURL"] = random.string(chars=string.ascii_letters)

    @classmethod
    def tearDownClass(cls):
        del os.environ["podUUID"]
        del os.environ["reporterGatewayURL"]

    def tearDown(self):
        report_promethues.push.failures = 0

class ReporterPromethuesPushTest(ReporterPromethuesBaseTest):
    def _push(self):
        args = random_args()
        report_promethues.push(*args)

    def testInitialization(self):
        self.assertEqual(report_promethues.push.failures, 0)

    def testSanity(self):
        with Mock() as mock:
            for i in range(random.number(2, 20)):
                self.assertEqual(mock.count, i)
                self._push()
                self.assertEqual(mock.count, i + 1)

    def testIOError(self):
        with Mock(IOError):
            try:
                self._push()
            except:
                self.fail('IOError was not excepted')

    def testNonIOError(self):
        for error in [ImportError, IndexError, KeyError, ValueError]:
            with Mock(error):
                with self.assertRaises(error):
                    self._push()

    def testErrorCached(self):
        with Mock(IOError) as mock:
            for i in range(report_promethues.RETRIES):
                self.assertEqual(mock.count, i)
                self._push()
                self.assertEqual(mock.count, i + 1)

            for _ in range(random.number(2, 20)):
                self._push()
                self.assertEqual(mock.count, report_promethues.RETRIES)

class ReporterPromethuesWorkerTest(ReporterPromethuesBaseTest):
    def testCreationManual(self):
        for daemon in [True, False]:
            worker = report_promethues.Worker()

            worker.start(daemon=daemon)
            self.assertTrue(pid_exists(worker.pid))
            worker.finish()
            self.assertFalse(pid_exists(worker.pid))

    def testCreationScope(self):
        with report_promethues.Worker() as worker:
            self.assertTrue(pid_exists(worker.pid))

        self.assertFalse(pid_exists(worker.pid))

    def testSend(self):
        with Mock():
            with report_promethues.Worker() as worker:
                worker.send(random_args())

    def testSendIOError(self):
        with Mock(IOError):
            with report_promethues.Worker() as worker:
                for _ in range(random.number(2, 20)):
                    worker.send(random_args())

    def testSendError(self):
        for error in [ImportError, IndexError, KeyError, ValueError]:
            with Mock(error):
                worker = report_promethues.Worker()
                worker.start(daemon=False)

                for _ in range(random.number(2, 20)):
                    worker.send(random_args())

                # `finish` should not fail even if the worker failed
                try:
                    worker.finish()
                except:
                    self.fail('`finish` was not expected to raise an error')

class ReporterPromethuesReportTest(ReporterPromethuesBaseTest):
    def tearDown(self):
        super(ReporterPromethuesReportTest, self).tearDown()
        report_promethues.WORKER = None

    def testInitialization(self):
        self.assertIsNone(report_promethues.WORKER)

    def testSanity(self):
        with Mock():
            self.assertIsNone(report_promethues.WORKER)
            report_promethues.report(*random_args())
            self.assertIsNotNone(report_promethues.WORKER)
            self.assertTrue(report_promethues.WORKER.daemon)

    def testFinish(self):
        for error in [None, ImportError, IndexError, KeyError, ValueError]:
            with Mock(error):
                for _ in range(random.number(2, 5)):
                    report_promethues.report(*random_args())

                    pid = report_promethues.WORKER.pid
                    self.assertTrue(pid_exists(pid))

                    # `finish` should not fail even if the worker failed
                    try:
                        report_promethues.finish()
                    except:
                        self.fail('`finish` was not expected to raise an error')

                    self.assertFalse(pid_exists(pid))
                    self.assertIsNone(report_promethues.WORKER)

if __name__ == '__main__':
    unittest.main()
