import numpy as np


class RunningMeter:
    def __init__(self, args):
        # Tracking at a per epoch level
        self.loss = {"train": [], "val": [], "test": []}
        self.accuracy = {"train": [], "val": [], "test": []}
        self.f1_score = {"train": [], "val": [], "test": []}
        self.f1_score_weighted = {"train": [], "val": [], "test": []}

        self.epochs = np.arange(0, args.num_epochs + 1)

        # Separated losses
        self.kmeans_loss = {"train": [], "val": [], "test": []}
        self.multi_loss = {"train": [], "val": [], "test": []}

        self.best_meter = BestMeter()

        self.args = args

    def update(
        self,
        phase,
        loss,
        kmeans_loss,
        multi_loss,
        accuracy,
        f1_score,
        f1_score_weighted,
    ):
        self.loss[phase].append(loss)
        self.kmeans_loss[phase].append(kmeans_loss)
        self.multi_loss[phase].append(multi_loss)
        self.accuracy[phase].append(accuracy)
        self.f1_score[phase].append(f1_score)
        self.f1_score_weighted[phase].append(f1_score_weighted)

    def get(self):
        return (
            self.loss,
            self.kmeans_loss,
            self.multi_loss,
            self.accuracy,
            self.f1_score,
            self.f1_score_weighted,
            self.epochs,
        )

    def update_best_meter(self, best_meter):
        self.best_meter = best_meter


class BestMeter:
    def __init__(self):
        # Storing the best values
        self.loss = {"train": np.inf, "val": np.inf, "test": np.inf}
        self.kmeans_loss = {"train": np.inf, "val": np.inf, "test": np.inf}
        self.multi_loss = {"train": np.inf, "val": np.inf, "test": np.inf}
        self.accuracy = {"train": 0.0, "val": 0.0, "test": 0.0}
        self.f1_score = {"train": 0.0, "val": 0.0, "test": 0.0}
        self.f1_score_weighted = {"train": 0.0, "val": 0.0, "test": 0.0}
        self.epoch = 0

        # For cross validation, we can track the test split predictions and
        # gt to compute the f1-score at the end
        self.preds = []
        self.gt = []

    def update(
        self,
        phase,
        loss,
        kmeans_loss,
        multi_loss,
        accuracy,
        f1_score,
        f1_score_weighted,
        epoch,
    ):
        self.loss[phase] = loss
        self.kmeans_loss[phase] = kmeans_loss
        self.multi_loss[phase] = multi_loss
        self.accuracy[phase] = accuracy
        self.f1_score[phase] = f1_score
        self.f1_score_weighted[phase] = f1_score_weighted
        self.epoch = epoch

    def get(self):
        return (
            self.loss,
            self.kmeans_loss,
            self.multi_loss,
            self.accuracy,
            self.f1_score,
            self.f1_score_weighted,
            self.epoch,
        )

    def display(self):
        print("The best epoch is {}".format(self.epoch))
        for phase in ["train", "val", "test"]:
            print(
                "Phase: {}, loss: {}, accuracy: {}, f1_score: {}, f1_score "
                "weighted: {}".format(
                    phase,
                    self.loss[phase],
                    self.kmeans_loss[phase],
                    self.multi_loss[phase],
                    self.accuracy[phase],
                    self.f1_score[phase],
                    self.f1_score_weighted[phase],
                )
            )
