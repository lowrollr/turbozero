


from core.test.tester import TwoPlayerTester


class ConnectXTester(TwoPlayerTester):
    def add_evaluation_metrics(self, episodes):
        if self.history is not None:
            for _ in episodes:
                self.history.add_evaluation_data({}, log=self.log_results)