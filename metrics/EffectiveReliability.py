import numpy as np
import pandas as pd
import torch
import os

NULL = np.inf


# Naive accuracy
class ClassificationVQAAccuracyEvaluator:
    def eval_pred_list(self, pred_answers, gt_answers, *args, **kwargs):
        return [int(pred_answer == gt_answer) for pred_answer, gt_answer in zip(pred_answers, gt_answers)]


class EffectiveReliability: #(BaseMetric):
    """
    Computes the Effective Reliability metric (Phi_c), calculated as follows,
    with variables:
      - x: input
      - g: selection function, with output in {0,1}. g(x)=0 indicates abstention on x,
           whereas g(x)=1 indicates an answer is given for x.
      - Acc: accuracy function; in this case, VQA Accuracy
      - c: cost value

    Phi_c(x) =   Acc(x),   if g(x) = 1 and Acc(x) > 0
                 -c,       if g(x) = 1 and Acc(x) = 0
                 0,        if g(x) = 0.

    The final Phi_c is a summation over Phi_c(x) for each sample x.

    EffectiveReliability.compute calculates the following:
    1. If the input `precomputed_cost_threshold_file` is None, computes the best possible
       thresholds for each of the cost values, and saves to a CSV file. Typically, this is
       done on a validation set, and the thresholds can then be used for the test set eval.
       If not None, the CSV file is loaded, and Phi_c is computed using the saved thresholds.

    2. Best possible Phi_c for each of the input cost values, as well as the
       corresponding risk and coverage (i.e., using the best possible g, which
       abstains only on samples where Acc(x) = 0).

    3. Phi_c for each of the input cost values, where g(x) is always 1 (as is the case for
       models which do not have the option to abstain).

    Initialization of this metric has the following arguments:
    - `cost_values`:
          List of numerical cost values `c` to use for separate Phi_c calculations
          e.g: [1, 10, 100]
    - `save_dir`:
          Directory to save CSV file to
    - `precomputed_cost_threshold_file`:
          If not None, path to CSV file with columns "Cost" (with cost values) and
          "Threshold" (with the corresponding thresholds).
    """
    def __init__(self, cost_values=[1, 10, 100], save_dir="saveEffectiveReliability", precomputed_cost_threshold_file=None, **kwargs):
        # super().__init__("effective_reliability")
        self.required_params = ["scores", "targets", "confidences", "__prediction_report__"]
        self.acc_evaluator = ClassificationVQAAccuracyEvaluator()
        self.cost_values = cost_values
        self.precomputed_cost_threshold_file = precomputed_cost_threshold_file
        self.save_dir = save_dir

    def _get_accuracies(self, pred_answers, gt_answers, *args, **kwargs):
        return self.acc_evaluator.eval_pred_list(pred_answers, gt_answers, *args, **kwargs)

    # def _broadcast_result(self, result_dict):
    #     broad_result = {}
    #     for k, t in result_dict.items():
    #         broad_result[k] = broadcast_tensor(t, src=0)
    #     return broad_result

    def _get_sorted_costs(self, sorted_scores):
        """
        Return dictionary mapping a cost value c to an array sorted_costs
        of the same length as sorted_scores, where each entry sorted_costs[i]
        contains the value phi_c[i], computed assuming sample i was NOT abstained on
        (i.e., g(x_i) = 1).
        """
        cost2sorted_costs = {}
        for c in self.cost_values:
            sorted_costs = []
            for s in sorted_scores:
                if s == 0:
                    sorted_costs.append(-c)
                else:
                    sorted_costs.append(s)
            cost2sorted_costs[c] = sorted_costs
        return cost2sorted_costs

    def _calc_best_possible_phi(self, sorted_costs):
        """
        Given an array with phi_c values computed without abstention,
        calculate the best possible phi_c (where g(x) = 0 iff. Acc(x) = 0,
        for each x).

        Compute the corresponding risk and coverage as well.
        """
        total_questions = len(sorted_costs)

        # Add up all positive entries of sorted_costs
        sorted_costs = np.array(sorted_costs)
        max_phi = sorted_costs[sorted_costs > 0].sum()
        best_possible_phi = max_phi / total_questions

        # Coverage
        num_answered = (sorted_costs > 0).sum()
        best_coverage = num_answered / total_questions

        # Risk
        # max_phi is the sum of Acc(x) scores on samples where Acc(x) > 0.
        # A perfect model gets Acc(x) = 1 each time, which equals num_answered,
        # giving a risk of 0.
        best_risk = 1 - (max_phi / num_answered)

        return best_possible_phi, best_coverage, best_risk

    def _calc_cost_threshold(self, sorted_confs, sorted_costs):
        """
        Given a list of model confidences and corresponding phi_c cost values
        computed without abstention, return the confidence threshold for abstention
        which maximizes phi_c.
        """
        all_phis = []
        for i in range(len(sorted_confs)):
            phi = sum(sorted_costs[:i])
            all_phis.append(phi)
        all_phis = np.array(all_phis)
        threshold_index = np.argmax(all_phis)
        threshold = sorted_confs[threshold_index]
        return threshold

    def _calc_phi_from_precomputed_threshold(
            self, c, threshold, sorted_confs, sorted_scores
    ):
        """
        Given a cost value c, threshold on model confidence,
        model confidences and associated accuracy scores,
        return phi_c and corresponding coverage and risk.
        """
        cum_score = 0.
        acc_score = 0.
        num_answered = 0
        total_questions = len(sorted_confs)
        for i in range(total_questions):
            if sorted_confs[i] > threshold:
                # Choose to answer
                acc_score += sorted_scores[i]
                num_answered += 1
                if sorted_scores[i] == 0:
                    cum_score -= c
                else:
                    cum_score += sorted_scores[i]
            else:
                # Choose to abstain; no updates to the score.
                pass
        phi = cum_score / total_questions
        coverage = num_answered / total_questions
        risk = 1 - (acc_score / num_answered)
        return phi, coverage, risk

    def _calc_phi_no_abstention(self, c, sorted_scores):
        """
        Given an array of accuracy scores, compute phi_c where the
        model must answer each question (i.e., g(x) = 1 for all samples x).
      	"""
        cum_score = 0
        for s in sorted_scores:
            # No option to abstain
            if s == 0:
                cum_score -= c
            else:
                cum_score += s
        phi_no_abstention = cum_score / len(sorted_scores)
        return phi_no_abstention

    def _calc_er_result(self, scores, confidences, device, best):
        assert len(scores) == len(confidences), \
            "{} != {}".format(len(scores), len(confidences))
        sorted_confs, sorted_scores = \
            zip(*sorted(
                [tup for tup in zip(confidences, scores)],
                key=lambda x: -x[int(best)]
            ))

        results = {}
        cost2sorted_costs = self._get_sorted_costs(sorted_scores)

        if best:
            # Compute best possible effective reliability score
            for c in self.cost_values:
                best_possible_phi, best_coverage, best_risk = \
                    self._calc_best_possible_phi(cost2sorted_costs[c])
                for key, val in [
                        (f'best_phi_c@{c}', best_possible_phi),
                        (f'best_cov_phi_c@{c}', best_coverage),
                        (f'best_risk_phi_c@{c}', best_risk)
                ]:
                    results[key] = torch.tensor(val, dtype=torch.float, device=device)
        else:
            if self.precomputed_cost_threshold_file is None:
                # Compute cost thresholds and save to a new file
                thresholds = []
                for c in self.cost_values:
                    threshold = self._calc_cost_threshold(
                        sorted_confs,
                        cost2sorted_costs[c],
                    )
                    thresholds.append((c, threshold))
                csv_path = 'thresholds_at_costs.csv'
                csv_path = os.path.join(self.save_dir, csv_path)
                print(f'Saving effective reliability cost threshold info to {csv_path}')
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                with open(csv_path, "w") as f:
                    headers = "Cost,Threshold"
                    f.write(headers + "\n")
                    for c, t in thresholds:
                        f.write("{},{}\n".format(c,t))
            else:
                # Load precomputed cost thresholds, and compute
                # effective reliability score
                threshold_data = pd.read_csv(self.precomputed_cost_threshold_file)
                costs = threshold_data['Cost'].values
                thresholds = threshold_data['Threshold'].values
                for c, threshold in zip(costs, thresholds):
                    phi, cov, risk = \
                        self._calc_phi_from_precomputed_threshold(
                            c, threshold, sorted_confs, sorted_scores
                        )
                    for key, val in [
                            (f'phi_c@{c}', phi),
                            (f'cov_phi_c@{c}', cov),
                            (f'risk_phi_c@{c}', risk)
                    ]:
                        results[key] = torch.tensor(
                            val, dtype=torch.float, device=device
                        )

                    # Compute effective reliability without abstention option
                    phi_no_abstention = self._calc_phi_no_abstention(c, sorted_scores)
                    key = f'no_abstention_phi_c@{c}'
                    results[key] = torch.tensor(
                        phi_no_abstention, dtype=torch.float, device=device
                    )

        return results

    def calculate(
        self, model_output, *args, **kwargs
        #self, sample_list, model_output, execute_on_master_only=True, *args, **kwargs
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu" #get_current_device()
        # keys = []

        # if self.precomputed_cost_threshold_file:
        #     threshold_data = pd.read_csv(self.precomputed_cost_threshold_file)
        #     costs = threshold_data['Cost'].values
        #     for c in costs:
        #         keys.append(f'phi_c@{c}')
        #         keys.append(f'cov_phi_c@{c}')
        #         keys.append(f'risk_phi_c@{c}')
        #         keys.append(f'no_abstention_phi_c@{c}')

        # for c in self.cost_values:
        #     keys.append(f'best_phi_c@{c}')
        #     keys.append(f'best_cov_phi_c@{c}')
        #     keys.append(f'best_risk_phi_c@{c}')

        # if execute_on_master_only: #and not is_master():
        #     results = OrderedDict()
        #     for key in keys:
        #         # dummy to be overridden in broadcasting
        #         results[key] = torch.tensor(NULL, dtype=torch.float, device=device)
        # else:
        output = []
        expected = []
        confidences = []

        for entry in model_output: #.prediction_report:
            output.append(entry["answer"])
            expected.append(entry["gt_answers"])
            confidences.append(entry["confidence"])

        acc_scores = self._get_accuracies(output, expected)

        results = self._calc_er_result(acc_scores, confidences, device, best=False)
        results.update(self._calc_er_result(acc_scores, confidences, device, best=True))

        # if execute_on_master_only:
        #      results = self._broadcast_result(results)
        return results


if __name__=="__main__":

    def generate_testcase(n, sigma=0.4):
        answers = np.random.randint(1, 11, size=n)
        
        gt_answers = np.round(np.random.normal(loc=answers, scale=sigma)).astype(int)
        gt_answers = np.clip(gt_answers, 1, 10)
        
        epsilon = np.random.poisson(10, n)/100
        confidences = np.exp(-np.abs(answers - gt_answers) - epsilon)

        noise_indices = np.random.choice(n, int(0.1*n), replace=False)
        for idx in noise_indices:
            if gt_answers[idx] != confidences[idx]:
                gt_answers[idx] = confidences[idx]
            else:
                gt_answers[idx] = confidences[idx] + 1

        print("Generating testcase ...")
        print("Number of true prediction =", sum(int(i == j) for i, j in zip(answers, gt_answers)))
        print("Expected true prediction confidence =", sum((k if i == j else 0) for i, j, k in zip(answers, gt_answers, confidences)) / sum(int(i == j) for i, j in zip(answers, gt_answers)))
        print("Expected false prediction confidence =", sum((k if i != j else 0) for i, j, k in zip(answers, gt_answers, confidences)) / sum(int(i != j) for i, j in zip(answers, gt_answers)))
        
        model_output = [{"answer": i,
                        "gt_answers": j,
                        "confidence": k}
                        for i, j, k in zip(answers, gt_answers, confidences)]

        return model_output

    er_evaluator = EffectiveReliability()

    model_output = generate_testcase(10000)

    results = er_evaluator.calculate(model_output)
    print(results)