import json
from evaluation.vqa import VQA
from evaluation.reliable_vqa_eval import ReliabilityEval


class EvaluatorModeHandler:
    def __init__(self, predict_df, threshold_df=None):
        self.predict_df = predict_df
        if threshold_df is None or threshold_df.empty:
            print(1)
            self.threshold_df = predict_df
        else:
            self.threshold_df = threshold_df

        self.predict_df['question_id'] = range(1, len(self.predict_df) + 1)
        self.threshold_df['question_id'] = range(1, len(self.threshold_df) + 1)

        self.questions = self._build_questions(self.predict_df)
        self.annotations = self._build_annotations(self.predict_df)
        self.predictions = self._build_predictions(self.predict_df)
        self.threshold_predictions = self._build_predictions(self.threshold_df)

    def _build_questions(self, df):
        return {
            'questions': [
                {
                    'image_id': row['img_id'],
                    'question': row['question'],
                    'question_id': row['question_id']
                }
                for _, row in df.iterrows()
            ]
        }

    def _build_annotations(self, df):
        return {
            'annotations': [
                {
                    'answers': row['ground_truth'],
                    'image_id': row['img_id'],
                    'question_id': row['question_id']
                }
                for _, row in df.iterrows()
            ]
        }

    def _build_predictions(self, df):
        return [
            {
                'question_id': row['question_id'],
                'answer': row['answer'],
                'confidence': row['confidence'],
                'image_id': row['img_id']
            }
            for _, row in df.iterrows()
        ]

    def _load_data_from_dicts(self, risk_tolerances):
        ann_vqa = VQA(annotations=self.annotations, questions=self.questions)
        all_qids = ann_vqa.getQuesIds()

        vqa_eval = ReliabilityEval(
            all_qids, risk_tolerances=risk_tolerances, n=2
        )

        res_vqa = ann_vqa.loadRes(VQA(), self.predictions)
        threshold_res_vqa = (
            ann_vqa.loadRes(VQA(), self.threshold_predictions)
            if self.threshold_df is not None else None
        )

        return ann_vqa, res_vqa, threshold_res_vqa, vqa_eval

    def evaluate(self, risk_tolerances=[0.01, 0.05, 0.1, 0.2]):
        gt_data, pred_data, threshold_pred_data, evaluator = self._load_data_from_dicts(
            risk_tolerances)

        qids = set(pred_data.getQuesIds())
        threshold_qids = set(threshold_pred_data.getQuesIds()
                             ) if threshold_pred_data else None

        evaluator.evaluate(
            gt_data,
            pred_data,
            threshold_pred_data,
            quesIds=qids,
            thresholdQuesIds=threshold_qids,
        )

        print(json.dumps(evaluator.accuracy, sort_keys=True, indent=4))
