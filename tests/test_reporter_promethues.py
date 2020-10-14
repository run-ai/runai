import os
import string
import unittest

import runai.utils
import runai.reporter

class Mock(runai.utils.Hook):
    def __init__(self, error=None):
        super(Mock, self).__init__(runai.reporter.report_promethues, 'pushadd_to_gateway')
        self._error = error
        self.count = 0

    def __hook__(self, *args, **kwargs):
        self.count += 1
        if self._error:
            raise self._error()

class ReportPromethuesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["podUUID"] = runai.utils.random.string()
        os.environ["reporterGatewayURL"] = runai.utils.random.string()

    def setUp(self):
        # reset the library cache
        runai.reporter.report_promethues.createGaugeAndPushToGateway.FAILED = False

    def _call(self):
        reporter_name = runai.utils.random.string(chars=string.ascii_letters + string.digits)
        reporter_value = runai.utils.random.number()
        report_type = runai.utils.random.choice([runai.reporter.report_promethues.ReportType.metric, runai.reporter.report_promethues.ReportType.parameter])

        runai.reporter.report_promethues.createGaugeAndPushToGateway(reporter_name, reporter_value, report_type)

    def testSanity(self):
        with Mock() as mock:
            for i in range(runai.utils.random.number(2, 20)):
                self.assertEqual(mock.count, i)
                self._call()
                self.assertEqual(mock.count, i + 1)

    def testIOError(self):
        with Mock(IOError):
            try:
                self._call()
            except:
                self.fail('IOError was not excepted')

    def testNonIOError(self):
        for error in [ImportError, IndexError, KeyError, ValueError]:
            with Mock(error):
                with self.assertRaises(error):
                    self._call()

    def testErrorCached(self):
        with Mock(IOError) as mock:
            self.assertEqual(mock.count, 0)
            self._call()
            self.assertEqual(mock.count, 1)

            for _ in range(runai.utils.random.number(2, 20)):
                self._call()
                self.assertEqual(mock.count, 1)

if __name__ == '__main__':
    unittest.main()
