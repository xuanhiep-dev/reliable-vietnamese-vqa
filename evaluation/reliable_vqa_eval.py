# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# LICENSE from https://github.com/GT-Vision-Lab/VQA
# Copyright (c) 2014, Aishwarya Agrawal
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are
# those
# of the authors and should not be interpreted as representing official
# policies,
# either expressed or implied, of the FreeBSD Project.

import re

from utils.dataset import Process
from collections import OrderedDict

import numpy as np
from sklearn.metrics import auc


class ReliabilityEval:
    def __init__(self, quesIds, risk_tolerances=None, costs=None, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = OrderedDict()
        self.evalThresholdQA = OrderedDict()
        self.processor = Process()
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.all_qids = quesIds

        self.risk_tolerances = risk_tolerances
        self.costs = costs

        if self.risk_tolerances is None:
            self.risk_tolerances = [0.01, 0.05, 0.1, 0.2]
        if self.costs is None:
            self.costs = [1, 10, 100]

        self.cost_thresholds = None

        self.fptp_stats = {}

    def evaluate(
        self,
        vqa,
        vqaRes,
        thresholdRes,
        quesIds=None,
        thresholdQuesIds=None,
        skip_reliability=False,
    ):
        if quesIds == None:
            quesIds = self.all_qids

        # =================================================
        # Compute accuracy
        # =================================================
        self.computeAccuracy(vqa, vqaRes, quesIds, is_threshold=False)

        if skip_reliability:
            return

        self.setCatR()

        if thresholdRes is not None:
            assert thresholdQuesIds is not None
            # Set self.evalThresholdQA with accuracy & confidence for each prediction in
            # the threshold set
            self.computeAccuracy(
                vqa, thresholdRes, thresholdQuesIds, is_threshold=True)

            # Compute thresholds for effective reliability, and update self.cost_thresholds
            self.computeThresholds()
            print("Chosen thresholds for effective reliability:")
            for cost, threshold in self.cost_thresholds.items():
                print("Cost: {}, Threshold: {}".format(cost, threshold))
            # Compute effective reliability scores for each cost value
            self.setPhiAtCosts()

    def computeAccuracy(self, vqa, vqaRes, quesIds=None, is_threshold=False):
        accQA = []

        for quesId in quesIds:
            gt = vqa.qa[quesId]
            res = vqaRes.qa[quesId]

            gtAnswer = gt["answers"]
            resAns = res["answer"]
            gtAnswer = gtAnswer.replace("\n", " ").replace("\t", " ").strip()
            resAns = resAns.replace("\n", " ").replace("\t", " ").strip()

            resConf = res["confidence"]

            gtAnswer = self.processor.process_punctuation(gtAnswer.lower())
            resAns = self.processor.process_punctuation(resAns.lower())

            #######################################################
            acc = 1 if gtAnswer == resAns else 0
            #######################################################
            risk = 1.0 - acc
            accQA.append(acc)
            ########################################################
            self.setEvalQA(quesId, acc, risk, resConf,
                           is_threshold=is_threshold)
            ########################################################

        if not is_threshold:
            self.setAccuracy(accQA)

    def setAccuracy(self, accQA):
        self.accuracy["vqa_accuracy"] = round(
            100 * float(sum(accQA)) / len(accQA), self.n
        )

    def setRiskCoverage(self, evalQA, best=False):
        total_questions = len(evalQA)

        covered = 0.0
        cum_score = 0.0

        risks = []
        coverages = []

        temp_items = list(evalQA.items())

        for i, (_, qresult) in enumerate(evalQA.items()):
            if i > 0 and not best:
                assert temp_items[i -
                                  1][1]["confidence"] >= qresult["confidence"]

            covered += 1.0
            cum_score += qresult["indiv_risk"]

            curr_risk = cum_score / covered
            curr_cov = covered / total_questions

            qresult["risk"] = curr_risk
            qresult["coverage"] = curr_cov

            risks.append(curr_risk)
            coverages.append(curr_cov)

        auc_score = auc(coverages, risks)

        key = "auc"
        if best:
            key = "best_" + key

        self.accuracy[key] = round(100.0 * auc_score, self.n)

    def computeCatR(self, evalQA, risk_tolerance, best=False):
        total_questions = len(evalQA)

        _, rc_data = zip(*evalQA.items())
        index = total_questions
        while index > 0 and rc_data[index - 1]["risk"] > risk_tolerance:
            index -= 1
        index -= 1

        if (
            -1 < index < (total_questions - 1)
            and rc_data[index]["confidence"] == rc_data[index + 1]["confidence"]
        ):
            while index > -1 and (
                rc_data[index]["confidence"] == rc_data[index + 1]["confidence"]
                or rc_data[index]["risk"] > risk_tolerance
            ):
                index -= 1

        cov = rc_data[index]["coverage"] if index > -1 else 0.0
        threshold = rc_data[index]["confidence"] if index > -1 else 0.0

        catr = {
            "coverage": round(100.0 * cov, self.n),
            "threshold": threshold,
        }

        key = "cov@{}".format(str(risk_tolerance))

        if best:
            key = "best_" + key

        self.accuracy[key] = catr

    def setCatR(self):
        for is_best in (True, False):
            sort_key = "confidence" if not is_best else "accuracy"

            self.evalQA = OrderedDict(
                sorted(self.evalQA.items(), key=lambda x: -x[1][sort_key])
            )

            self.setRiskCoverage(self.evalQA, best=is_best)

            for rt in self.risk_tolerances:
                self.computeCatR(self.evalQA, rt, best=is_best)

        self.evalQA = OrderedDict(
            sorted(self.evalQA.items(), key=lambda x: -x[1]["confidence"])
        )

    def setPhiAtCosts(self):
        sorted_confs, sorted_scores = self.getSortedArraysForPhi(
            is_threshold=False)
        for cost in self.costs:
            self.computePhiAtCost(sorted_confs, sorted_scores, cost)
            self.computeBestPossiblePhiAtCost(sorted_scores, cost)

    def getSortedArraysForPhi(self, is_threshold=False):
        qa = self.evalThresholdQA if is_threshold else self.evalQA
        qa = OrderedDict(sorted(qa.items(), key=lambda x: -x[1]["confidence"]))
        sorted_confs = [r["confidence"] for r in qa.values()]  # High to low
        sorted_scores = [
            r["accuracy"] / 100 for r in qa.values()
        ]  # Corresponding scores
        sorted_confs = np.array(sorted_confs)
        sorted_scores = np.array(sorted_scores)
        return sorted_confs, sorted_scores

    def computePhiAtCost(self, sorted_confs, sorted_scores, cost, prefix=""):
        threshold = self.cost_thresholds[cost]
        cum_score = 0.0
        acc_score = 0.0
        num_answered = 0
        total_questions = len(sorted_confs)
        for i in range(total_questions):
            if sorted_confs[i] >= threshold:
                # Choose to answer
                acc_score += sorted_scores[i]
                num_answered += 1
                if sorted_scores[i] == 0:
                    cum_score -= cost
                else:
                    cum_score += sorted_scores[i]

        phi = cum_score / total_questions
        cov = num_answered / total_questions
        risk = 1 - (acc_score / max(num_answered, 1.0))

        # Compute phi without option to abstain
        cum_score = 0.0
        for s in sorted_scores:
            # No option to abstain
            if s == 0:
                cum_score -= cost
            else:
                cum_score += s
        phi_no_abstention = cum_score / len(sorted_scores)

        res = {
            "threshold": threshold,
            "phi": round(100.0 * phi, self.n),
            "coverage": round(100.0 * cov, self.n),
            "risk": round(100.0 * risk, self.n),
            "no_abstention_phi": round(100.0 * phi_no_abstention, self.n),
        }
        self.accuracy[f"phi@{str(cost)}"] = res

    def computeBestPossiblePhiAtCost(self, sorted_scores, cost):
        sorted_costs = []
        for s in sorted_scores:
            if s == 0:
                sorted_costs.append(-cost)
            else:
                sorted_costs.append(s)
        sorted_costs = np.array(sorted_costs)
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
        best_risk = 1 - (max_phi / max(num_answered, 1.0))

        res = {
            "phi": round(100.0 * best_possible_phi, self.n),
            "coverage": round(100.0 * best_coverage, self.n),
            "risk": round(100.0 * best_risk, self.n),
        }
        self.accuracy[f"best_phi@{str(cost)}"] = res

    def computeThresholds(self):
        self.cost_thresholds = {}
        sorted_confs, sorted_scores = self.getSortedArraysForPhi(
            is_threshold=True)
        for c in self.costs:
            self.cost_thresholds[c] = self.computeThresholdAtCost(
                sorted_confs, sorted_scores, c
            )

    def computeThresholdAtCost(self, sorted_confs, sorted_scores, c):
        sorted_costs = []
        for score in sorted_scores:
            if score == 0.0:
                sorted_costs.append(-c)
            else:
                sorted_costs.append(score)
        sorted_costs = np.array(sorted_costs)

        all_phis = []
        for i in range(len(sorted_confs)):
            phi = sum(sorted_costs[: i + 1])
            all_phis.append(phi)
        all_phis = np.array(all_phis)
        threshold_candidates = np.where(all_phis == all_phis.max())[0]
        threshold_index = threshold_candidates[
            -1
        ]  # Lowest threshold (i.e., most coverage) with max phi
        threshold = sorted_confs[threshold_index]
        return threshold

    def setEvalQA(self, quesId, acc, risk, conf, is_threshold=False):
        qa = self.evalThresholdQA if is_threshold else self.evalQA
        qa[quesId] = {
            "accuracy": round(100.0 * acc, self.n),
            "indiv_risk": risk,
            "confidence": float(conf),
        }
