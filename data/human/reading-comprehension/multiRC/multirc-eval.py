### Evaluation script used for evaluation of baselines for MultiRC dataset
# The evaluation script expects the questions, and predicted answers from separate json files.
# The predicted answers should be 1s and 0s (no real-valued scores)

import math
from functools import reduce
class Measures:

    @staticmethod
    def per_question_metrics(dataset, output_map):
        P = []
        R = []
        for p in dataset:
            for qIdx, q in enumerate(p["paragraph"]["questions"]):
                id = p["id"] + "==" + str(qIdx)
                if (id in output_map):
                    predictedAns = output_map.get(id)
                    correctAns = [int(a["isAnswer"]) for a in q["answers"]]
                    predictCount = sum(predictedAns)
                    correctCount = sum(correctAns)
                    assert math.ceil(sum(predictedAns)) == sum(predictedAns), "sum of the scores: " + str(sum(predictedAns))
                    agreementCount = sum([a * b for (a, b) in zip(correctAns, predictedAns)])
                    p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
                    r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
                    P.append(p1)
                    R.append(r1)
                else:
                    print("The id " + id + " not found . . . ")

        pAvg = Measures.avg(P)
        rAvg = Measures.avg(R)
        f1Avg = 2 * Measures.avg(R) * Measures.avg(P) / (Measures.avg(P) + Measures.avg(R))
        return [pAvg, rAvg, f1Avg]

    @staticmethod
    def exact_match_metrics(dataset, output_map, delta):
        EM = []
        for p in dataset:
            for qIdx, q in enumerate(p["paragraph"]["questions"]):
                id = p["id"] + "==" + str(qIdx)
                if (id in output_map):
                    predictedAns = output_map.get(id)
                    correctAns = [int(a["isAnswer"]) for a in q["answers"]]
                    print('predictedAns: ' , predictedAns)
                    print('correctAns: ' , correctAns)
                    
                    em = 1.0 if sum([abs(i - j) for i, j in zip(correctAns, predictedAns)]) <= delta  else 0.0
                    EM.append(em)
                else:
                    print("The id " + id + " not found . . . ")

        return Measures.avg(EM)

    @staticmethod
    def per_dataset_metric(dataset, output_map):
        agreementCount = 0
        correctCount = 0
        predictCount = 0
        for p in dataset:
            for qIdx, q in enumerate(p["paragraph"]["questions"]):
                id = p["id"] + "==" + str(qIdx)
                if (id in output_map):
                    predictedAns = output_map.get(id)
                    correctAns = [int(a["isAnswer"]) for a in q["answers"]]
                    predictCount += sum(predictedAns)
                    correctCount += sum(correctAns)
                    agreementCount += sum([a * b for (a, b) in zip(correctAns, predictedAns)])
                else:
                    print("The id " + id + " not found . . . ")
        print('predictCount: ', predictCount)
        print('correctCount: ', correctCount)
        print('agreementCount: ', agreementCount)

        p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        return [p1, r1, 2 * r1 * p1 / (p1 + r1)]

    @staticmethod
    def avg(l):
        return reduce(lambda x, y: x + y, l) / len(l)


import json

# this is the location of your data; has to be downloaded from http://cogcomp.org/multirc/
inputFile = 'data/machine/reading-comprehension/multiRC/dev_83-fixedIds.json'

measures = Measures()

def main():
    eval('data/human/reading-comprehension/multiRC/human-01.json')
    # eval('baseline-scores/allOnes.json')
    # eval('baseline-scores/allZeros.json')
    # eval('baseline-scores/simpleLR.json')
    # eval('baseline-scores/lucene_world.json')
    # eval('baseline-scores/lucene_paragraphs.json')

# the input to the `eval` function is the file which contains the binary predictions per question-id
def eval(outFile):
    input = json.load(open(inputFile))
    output = json.load(open(outFile))
    output_map = dict([[a["pid"] + "==" + a["qid"], a["scores"]] for a in output])

    assert len(output_map) == len(output), "You probably have redundancies in your keys"

    '''[P1, R1, F1m] = measures.per_question_metrics(input["data"], output_map)
    print("Per question measures (i.e. precision-recall per question, then average) ")
    print("\tP: " + str(P1) + " - R: " + str(R1) + " - F1m: " + str(F1m))'''
    EM0 = measures.exact_match_metrics(input["data"], output_map, 0)
    EM1 = measures.exact_match_metrics(input["data"], output_map, 1)
    print("\tEM0: " + str(EM0))
    print("\tEM1: " + str(EM1))
    [P2, R2, F1a] = measures.per_dataset_metric(input["data"], output_map)

    print("Dataset-wide measures (i.e. precision-recall across all the candidate-answers in the dataset) ")
    print("\tP: " + str(P2) + " - R: " + str(R2) + " - F1a: " + str(F1a))

if __name__ == "__main__":
    main()