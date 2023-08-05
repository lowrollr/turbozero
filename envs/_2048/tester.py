


from core.test.tester import Tester


class _2048Tester(Tester):
    def add_evaluation_metrics(self, episodes):
        if self.history is not None:
            for episode in episodes:
                moves = len(episode)
                last_state = episode[-1][0]
                high_square = 2 ** int(last_state.max().item())
                self.history.add_evaluation_data({
                    'reward': moves,
                    'high_square': high_square,
                }, log=self.log_results)