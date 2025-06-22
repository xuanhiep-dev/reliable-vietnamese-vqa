import copy
import datetime
import json


class VQA:
    def __init__(self, annotations=None, questions=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotations == None and not questions == None:
            print("loading VQA annotations and questions into memory...")
            time_t = datetime.datetime.utcnow()
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = annotations
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques
        print("index created!")

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def getQuesIds(self, imgIds=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param  imgIds    (int array)   : get question ids for given imgs
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if not len(imgIds) == 0:
            anns = sum(
                [self.imgToQA[imgId]
                    for imgId in imgIds if imgId in self.imgToQA],
                [],
            )
        else:
            anns = self.dataset["annotations"]

        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]

        if not len(quesIds) == 0:
            anns = sum(
                [self.qa[quesId]
                    for quesId in quesIds if quesId in self.qa], []
            )
        else:
            anns = self.dataset["annotations"]
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            print("Question: {}".format(self.qqa[quesId]["question"]))
            for ans in ann["answers"]:
                print("Answer {}: {}".format(ans["answer_id"], ans["answer"]))

    def loadRes(self, res, resFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res.questions = self.questions

        print("Loading and preparing results...     ")
        time_t = datetime.datetime.utcnow()
        if isinstance(resFile, str):
            anns = json.load(open(resFile))
        else:
            anns = resFile

        assert type(anns) == list, "results is not an array of objects"

        for ann in anns:
            quesId = ann["question_id"]
            qaAnn = self.qa[quesId]
            ann["image_id"] = qaAnn["image_id"]

        print(
            "DONE (t=%0.2fs)" % (
                (datetime.datetime.utcnow() - time_t).total_seconds())
        )

        res.dataset["annotations"] = anns
        res.createIndex()
        return res
