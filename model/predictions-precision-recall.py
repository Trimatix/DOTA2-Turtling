"""Measure precision, recall, average precision and f1 for predictions in a single CSV file.
Give a path to your CSV in argv.

Written by jasper law
"""


from pandas import read_csv
from matplotlib import pyplot as plt
import random
# from matplotlib import colors as mcolors
import sys

from tensorflow.python.types.core import Value

if len(sys.argv) < 2:
    raise ValueError("give csv")

colours = ["red","green","blue","orange","black","purple","brown"]
usedcolours = set()
def randomColour():
    # return random.choice(list(mcolors.XKCD_COLORS.keys()))
    if len(usedcolours) == len(colours):
        raise ValueError("All colours used")
    col = random.choice(colours)
    while col in usedcolours:
        col = random.choice(colours)
    usedcolours.add(col)
    return col
    # return random.choice(list(mcolors.XKCD_COLORS.keys()))

df = read_csv(sys.argv[1], header=0)
NORM_HERO_IDS = False

if NORM_HERO_IDS:
    ROOT_HERO_IDS = {(2 ** i) / (2 ** 129): i for i in range(1,130)}
else:
    ROOT_HERO_IDS = {2 ** i: i for i in range(1,130)}

thresholds = [i/100 for i in range(5,105,5)]
precs = []
recs = []
f1s = []
fps = []
tps = []
fns = []
tns = []

# TURTLE_THRESH = 0.15
for TURTLE_THRESH in thresholds:

    total_predictions = len(df) * 10

    total_positive_groundtruth = sum(len(df.loc[df[f"isTurtling{heroNum}"] == 1]) for heroNum in range(10))
    total_negative_groundtruth = sum(len(df.loc[df[f"isTurtling{heroNum}"] == 0]) for heroNum in range(10))

    total_positive_predictions = sum(len(df.loc[df[f"predictedIsTurtling{heroNum}"] > TURTLE_THRESH]) for heroNum in range(10))
    total_negative_predictions = sum(len(df.loc[df[f"predictedIsTurtling{heroNum}"] < TURTLE_THRESH]) for heroNum in range(10))

    correct_positive_predictions = 0
    correct_negative_predictions = 0

    incorrect_positive_predictions = 0
    incorrect_negative_predictions = 0

    for heroNum in range(10):
        posPredicts = df.loc[df[f"predictedIsTurtling{heroNum}"] > TURTLE_THRESH]
        negPredicts = df.loc[df[f"predictedIsTurtling{heroNum}"] < TURTLE_THRESH]

        correct_positive_predictions += len(posPredicts.loc[posPredicts[f"isTurtling{heroNum}"] == 1])
        incorrect_positive_predictions += len(posPredicts.loc[posPredicts[f"isTurtling{heroNum}"] == 0])

        correct_negative_predictions += len(negPredicts.loc[negPredicts[f"isTurtling{heroNum}"] == 0])
        incorrect_negative_predictions += len(negPredicts.loc[negPredicts[f"isTurtling{heroNum}"] == 1])

    # # print(f"{incorrect} incorrect predictions at {TURTLE_THRESH*100}% threshold")
    # print(f"Threshold: {TURTLE_THRESH * 100}")
    # print(f"correct positive predictions: {correct_positive_predictions}")
    # print(f"correct negative predictions: {correct_negative_predictions}")
    # print(f"incorrect positive predictions: {incorrect_positive_predictions}")
    # print(f"incorrect negative predictions: {incorrect_negative_predictions}")
    # print(f"precision: {round((correct_positive_predictions / total_positive_predictions) * 100, 2)}")
    # print(f"recall: {round((correct_positive_predictions / total_positive_groundtruth) * 100, 2)}")

    fps.append(incorrect_positive_predictions)
    tps.append(correct_positive_predictions)
    fns.append(incorrect_negative_predictions)
    tns.append(correct_negative_predictions)

    prec1 = correct_positive_predictions / max(total_positive_predictions, 1)
    rec1 = correct_positive_predictions / max(total_positive_groundtruth, 1)
    precs.append(prec1)
    recs.append(rec1)
    denom = (prec1+rec1) or 1
    f1s.append(2*((prec1*rec1)/denom))

print("average prec %.7f" % (sum(precs)/len(precs)))
print("average f1 %.7f" % (sum(f1s)/len(f1s)))
print("max prec %.7f" % max(precs))
print("max rec %.7f" % max(recs))
print("max f1 %.7f" % max(f1s))
print("average fps %.7f" % (sum(fps)/len(fps)))
print(f"fps range ({min(fps):.7f}, {max(fps):.7f})")
print("average tps %.7f" % (sum(tps)/len(tps)))
print(f"tps range ({min(tps):.7f}, {max(tps):.7f})")
print("average fns %.7f" % (sum(fns)/len(fns)))
print(f"fns range ({min(fns):.7f}, {max(fns):.7f})")
print("average tns %.7f" % (sum(tns)/len(tns)))
print(f"tns range ({min(tns):.7f}, {max(tns):.7f})")

plt.subplot(2,2,1)
plt.plot(recs, precs)
plt.xlabel("recall")
plt.ylabel("precision")
plt.xlim(0, 1)
plt.ylim(0, 1)
# plt.show()

# plt.clf()
plt.subplot(2,2,2)
plt.plot(thresholds, f1s)
plt.xlabel("threshold")
plt.ylabel("f1")
plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.show()

# plt.clf()
plt.subplot(2,2,3)
plt.plot(thresholds, tps, label="tps", color=randomColour())
plt.plot(thresholds, fps, label="fps", color=randomColour())
plt.plot(thresholds, tns, label="tns", color=randomColour())
plt.plot(thresholds, fns, label="fns", color=randomColour())
plt.xlabel("threshold")
plt.ylabel("predictions")
plt.legend(loc="upper right")
# plt.xlim(0, 1)
# plt.ylim(0, 40000)
plt.show()
