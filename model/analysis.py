"""Huge multi-tool for analysing players' and teams' statistics throughout a parsed match.
Used to generate all of the dataset analysis graphs in my diss.

Each plot also includes estimated rates of change based on power series curves of best fit for your data

CSV headers must contain tick and any hero attributes as <attributeName><heroNum> e.g kills0 (heroNum must be between 0 and 9 inclusive)
heroName and heroTeam are required columns for every hero (i.e heroName0 and heroTeam0 -> heroName9 and heroTeam9 must be present)

python analysis.py data.csv plot <statName> to plot totals for each team across time
python analysis.py data.csv cmp <statName> to plot the radiant advantage for statName across time, dotabuff style
python analysis.py data.csv hero <statName> <heroNum> to plot hero #heroNum's statName across time, compared to the average for the rest of the team
python analysis.py data.csv cmpTeam <statName> <teamName> to plot statName across time for all heroes on the given team, and by default also plot the team's average
python analysis.py data.csv heatmap <statName1> <statName2> <heroNum> to plot a heatmap of statName1 against statName2 for hero #heroNum. useful for plotting their positioning throughout the game, or how their stats change throughout the game E.g showing the relationship between last hits and net worth, but also how much time the hero spent at those values

may have forgotten some

modify the function calls on lines 212 - 217 for more behaviour, e.g:
- Change any function's degree for the 'degree' (accuracy) of the curve of best fit
    - There's also dirDegree for plotTeamsStat if you want Dire to have a different degree to radiant, and teamDegree for cmpHeroTeamStat to use different degrees for the chosen hero and team average
- Toggle showRaw/showPoly/showDeriv for plots of the raw data/curves of best fit/estimated rates of change
- Toggle average on cmpHeroTeamStat to use the team average or team total for your stat
- Toggle inclusive on cmpHeroTeamStat to include your hero in the average/total calculations
- Toggle rescalePolys on cmpHeroTeamStat or plotTeamStat to put curves of best fit onto the same scale (only really useful for comparing gradients)
- Customise grid resolution (bins), colours, and colour quantizing for heatmap
- rescaleDerivs options for all plots, scaling derivative plots up to the same scale as the raw data plots so they're more readable

written by jasper law
"""



from pandas import read_csv
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import random
import glob

if len(sys.argv) > 3:
    filePath = os.path.normpath(sys.argv[1])
    action = sys.argv[2].lower()
    stat = sys.argv[3]
else:
    raise ValueError("Please provide a path to your csv")

if action == "datafreqs":
    """Show the frequency distribution for a given feature
    """
    SHOW_AVERAGE = False
    REMOVE_ZERO = False
    VERT_XLABELS = False

    if os.path.isfile(filePath):
        files = [filePath]
    elif os.path.isdir(filePath):
        files = glob.glob(os.path.join(filePath,"*.csv"))
    else:
        raise FileNotFoundError("Unrecogised: " + filePath)

    if not files:
        raise FileNotFoundError("The requested directory is empty: " + filePath)
    else:
        df = read_csv(files[0], header=0)
        heroAtt = (stat + "0") in df.columns
        meanAverage = {}
        xFreqs = {}
        for pathNum, currentPath in enumerate(files):
            if len(files) > 1:
                print("(" + str(pathNum+1) + "/" + str(len(files)) + ") Reading " + os.path.split(currentPath)[1] + " ...")
            df = read_csv(currentPath, header=0)
            for line in range(df.shape[0]):
                for ext in tuple(str(i) for i in range(10)) if heroAtt else ("",):
                    currentVal = df[stat + ext][line]
                    if (not REMOVE_ZERO) or (REMOVE_ZERO and currentVal != 0):
                        if SHOW_AVERAGE and currentVal not in meanAverage:
                            meanAverage[currentVal] = 0
                        if currentVal not in xFreqs:
                            xFreqs[currentVal] = 0

                        if SHOW_AVERAGE:
                            meanAverage[currentVal] += currentVal
                        xFreqs[currentVal] += 1

        if SHOW_AVERAGE:
            for currentVal in meanAverage:
                meanAverage[currentVal] /= xFreqs[currentVal]

        plt.scatter(xFreqs.keys(), xFreqs.values(), label="Value frequencies")
        if SHOW_AVERAGE:
            plt.plot(meanAverage.keys(), meanAverage.values(), color="r", label="Mean average")

        if VERT_XLABELS:
            if stat == "heroName":
                plt.xticks(list(xFreqs.keys()), list(i.split("CDOTA_Unit_Hero_")[1] for i in xFreqs.keys()), rotation='vertical')
            else:
                plt.xticks(list(xFreqs.keys()), list(xFreqs.keys()), rotation='vertical')
        plt.xlabel(stat)
        plt.ylabel("frequency")
        if SHOW_AVERAGE:
            plt.legend(loc='upper left')

        # plt.legend(loc='upper left')
        plt.show()

    exit()

dataframe = read_csv(filePath, header=0)
unitY = np.linspace(0, 1, dataframe.shape[0])

def randomColour():
    """Pick a random colour to plot a line with
    """
    # c = random.choice(mcolors.XKCD_COLORS)
    # while c == "blue":
    #     c = random.choice(mcolors.XKCD_COLORS)
    return random.choice(list(mcolors.XKCD_COLORS.keys()))

def cmpMapControls():
    global dataframe
    global unitY
    plt.clf()
    dataframe = dataframe.sort_values("mapControlRadiant")
    plt.scatter(dataframe["mapControlRadiant"], dataframe["mapControlDire"], s=1)
    plt.xlabel("mapControlRadiant")
    plt.ylabel("mapControlDire")
    plt.show()

def plotMapControls():
    global dataframe
    global unitY
    plt.clf()
    dataframe = dataframe.sort_values("tick")
    d = plt.scatter(dataframe["tick"], dataframe["mapControlDire"], s=1)
    r = plt.scatter(dataframe["tick"], dataframe["mapControlRadiant"], color=randomColour(), s=1)
    plt.xlabel("tick")
    plt.ylabel("mapControlDire")
    # plt.legend(d, "dire map control")
    # plt.legend(r, "radiant map control")
    plt.show()





def plotTeamsStat(statName, polyDegree, radCol='g', dirCol='b', showPolys=True, rescalePolys=False, rescaleDerivs=False, dirDegree=None, showDeriv=True, showRaw=True):
    global dataframe
    global unitY
    dirDegree = dirDegree or polyDegree
    plt.clf()
    dataframe = dataframe.sort_values("tick")
    radGolds = np.ndarray(shape=(dataframe.shape[0],))
    dirGolds = np.ndarray(shape=(dataframe.shape[0],))
    strRadHeroNums = tuple(str(hero) for hero in range(10) if dataframe["heroTeam" + str(hero)][0] == "radiant")
    strDirHeroNums = tuple(str(hero) for hero in range(10) if dataframe["heroTeam" + str(hero)][0] == "dire")

    for lineNum in range(dataframe.shape[0]):
        radGolds[lineNum] = sum(dataframe[statName + hero][lineNum] for hero in strRadHeroNums)
        dirGolds[lineNum] = sum(dataframe[statName + hero][lineNum] for hero in strDirHeroNums)
    
    if showRaw:
        radPlot = plt.plot(dataframe["tick"], radGolds, color=radCol, label="Radiant")
        dirPlot = plt.plot(dataframe["tick"], dirGolds, color=dirCol, label="Dire")
    plt.xlabel("tick")
    plt.ylabel("total " + statName)
    radPoly = np.polynomial.Polynomial.fit(dataframe["tick"], radGolds, polyDegree)
    radPolyXs, radPolyYs = radPoly.linspace()
    radPolyDerivXs, radPolyDerivYs = radPoly.deriv().linspace()
    dirPoly = np.polynomial.Polynomial.fit(dataframe["tick"], dirGolds, dirDegree)
    dirPolyXs, dirPolyYs = dirPoly.linspace()
    dirPolyDerivXs, dirPolyDerivYs = dirPoly.deriv().linspace()
    if showPolys:
        if rescalePolys:
            radPolyYs = (radPolyYs / max(radPolyYs)) * max(radGolds)
            dirPolyYs = (dirPolyYs / max(dirPolyYs)) * max(radGolds)
        radPolyLine = plt.plot(dirPolyXs, dirPolyYs, ":", color=dirCol, label="Dire curve of best fit")
        radPolyLine = plt.plot(radPolyXs, radPolyYs, ":", color=radCol, label="Radiant curve of best fit")
    if showDeriv:
        if rescaleDerivs:
            dirPolyDerivYs = (dirPolyDerivYs / max(radPolyDerivYs)) * max(radGolds)
            radPolyDerivYs = (radPolyDerivYs / max(radPolyDerivYs)) * max(radGolds)

        radPolyDerivLine = plt.plot(radPolyDerivXs, radPolyDerivYs, "--", color=radCol, label="Estimated rate of change")
        dirPolyDerivLine = plt.plot(dirPolyDerivXs, dirPolyDerivYs, "--", color=dirCol, label="Estimated rate of change")

    plt.legend(loc='upper left')
    plt.show()

def cmpTeamsStat(statName, polyDegree, col='g', showPoly=True, showRaw=True, showDeriv=True, rescaleDerivs=False):
    global dataframe
    global unitY
    plt.clf()
    dataframe = dataframe.sort_values("tick")
    strRadHeroNums = tuple(str(hero) for hero in range(10) if dataframe["heroTeam" + str(hero)][0] == "radiant")
    strDirHeroNums = tuple(str(hero) for hero in range(10) if dataframe["heroTeam" + str(hero)][0] == "dire")
    # print("RAD HAS",len(strRadHeroNums),"HEROES")
    # print("DIR HAS",len(strRadHeroNums),"HEROES")
    # print("Radiant:",", ".join(dataframe["heroName" + heroNum][0] for heroNum in strRadHeroNums))
    # print("Dire:",", ".join(dataframe["heroName" + heroNum][0] for heroNum in strDirHeroNums))
    totalGolds = np.ndarray(shape=(dataframe.shape[0],))

    for lineNum in range(dataframe.shape[0]):
        totalGolds[lineNum] = sum(dataframe[statName + hero][lineNum] for hero in strRadHeroNums) - sum(dataframe[statName + hero][lineNum] for hero in strDirHeroNums)
    
    zeroPlot = plt.plot(dataframe["tick"], np.zeros(shape=totalGolds.shape), label="No advantage")

    if showRaw:
        goldsPlot = plt.plot(dataframe["tick"], totalGolds, color=col, label="Radiant " + statName + " advantage")
    plt.xlabel("tick")
    plt.ylabel("Radiant " + statName + " advantage")
    goldsPoly = np.polynomial.Polynomial.fit(dataframe["tick"], totalGolds, polyDegree)
    goldsPolyXs, goldsPolyYs = goldsPoly.linspace()
    goldsPolyDerivXs, goldsPolyDerivYs = goldsPoly.deriv().linspace()
    if showPoly:
        goldsPolyLine = plt.plot(goldsPolyXs, goldsPolyYs, ":", color=col, label="Curve of best fit")
    if showDeriv:
        if rescaleDerivs:
            goldsPolyDerivYs = (goldsPolyDerivYs / max(goldsPolyDerivYs)) * max(totalGolds)
        goldsPolyDerivLine = plt.plot(goldsPolyDerivXs, goldsPolyDerivYs, "--", color=col, label="Estimated rate of change")
    
    plt.legend(loc='lower left')
    plt.show()

def cmpHeroTeamStat(statName, heroNum, inclusive, polyDegree, heroCol='g', teamCol='b', showPolys=True, rescalePolys=False, rescaleDerivs=False, teamDegree=None, average=False, showDeriv=True):
    global dataframe
    global unitY
    teamDegree = teamDegree or polyDegree
    plt.clf()
    dataframe = dataframe.sort_values("tick")
    heroGolds = np.ndarray(shape=(dataframe.shape[0],))
    teamGolds = np.ndarray(shape=(dataframe.shape[0],))
    team = dataframe["heroTeam" + str(heroNum)][0]
    heroName = dataframe["heroName" + str(heroNum)][0].split("CDOTA_Unit_Hero_")
    heroName = heroName[1] if len(heroName) > 1 else heroName[0]
    strTeamHeroNums = tuple(str(hero) for hero in range(10) if dataframe["heroTeam" + str(hero)][0] == team)
    # enemyTeam = {"dire": "radiant", "radiant": "dire"}[team]
    for lineNum in range(dataframe.shape[0]):
        heroGolds[lineNum] = dataframe[statName + str(heroNum)][lineNum]
        teamGolds[lineNum] = sum(dataframe[statName + hero][lineNum] for hero in strTeamHeroNums)
        if not inclusive:
            teamGolds[lineNum] -= heroGolds[lineNum]
        if average:
            teamGolds[lineNum] /= (10 if inclusive else 9)
    heroPlot = plt.plot(dataframe["tick"], heroGolds, color=heroCol, label=heroName + " " + statName)
    teamPlot = plt.plot(dataframe["tick"], teamGolds, color=teamCol, label=team + (" average " if average else " total ") + statName + (("(" + heroName + " included)") if inclusive else ""))
    plt.xlabel("tick")
    plt.ylabel("total " + statName)
    # plt.legend(heroPlot, f"hero{heroNum!s}")
    # plt.legend(teamPlot, "whole team " + ("including" if inclusive else "excluding") + f" hero {heroNum!s}")
    heroPoly = np.polynomial.Polynomial.fit(dataframe["tick"], heroGolds, polyDegree)
    heroPolyXs, heroPolyYs = heroPoly.linspace()
    heroPolyDerivXs, heroPolyDerivYs = heroPoly.deriv().linspace()
    teamPoly = np.polynomial.Polynomial.fit(dataframe["tick"], teamGolds, teamDegree)
    teamPolyXs, teamPolyYs = teamPoly.linspace()
    teamPolyDerivXs, teamPolyDerivYs = teamPoly.deriv().linspace()

    if showPolys:
        if rescalePolys:
            heroPolyYs = (heroPolyYs / max(heroPolyYs)) * max(heroGolds)
            teamPolyYs = (teamPolyYs / max(teamPolyYs)) * max(heroGolds)
        heroPolyLine = plt.plot(heroPolyXs, heroPolyYs, ":", color=heroCol, label="Curve of best fit")
        teamPolyLine = plt.plot(teamPolyXs, teamPolyYs, ":", color=teamCol, label="Curve of best fit")

    if showDeriv:
        if rescaleDerivs:
            teamPolyDerivYs = (teamPolyDerivYs / max(heroPolyDerivYs)) * max(heroGolds)
            heroPolyDerivYs = (heroPolyDerivYs / max(heroPolyDerivYs)) * max(heroGolds)

        heroPolyDerivLine = plt.plot(heroPolyDerivXs, heroPolyDerivYs, "--", color=heroCol, label="Estimated rate of change")
        teamPolyDerivLine = plt.plot(teamPolyDerivXs, teamPolyDerivYs, "--", color=teamCol, label="Estimated rate of change")
    
    plt.legend(loc='upper left')
    plt.show()


def cmpWholeTeamStat(statName, teamName, polyDegrees=[3, 3, 3, 3, 3], averagePolyDegree=3, heroCols=['g', 'b', 'r', 'c', 'm'], averageCol='y', showRaw=True, showPolys=False, rescalePolys=False, rescaleDerivs=True, showDerivs=False, showAverage=True, rescaleAverage=False, showAveragePoly=False, rescaleAveragePoly=False, showAverageDeriv=False, rescaleAverageDeriv=True):
    global dataframe
    global unitY
    plt.clf()
    dataframe = dataframe.sort_values("tick")
    currentHero = 0
    if (showPolys or showDerivs) and (rescalePolys or rescaleDerivs):
        rescaleCoef = max(max(dataframe[statName + str(heroNum)]) for heroNum in range(10) if dataframe["heroTeam" + str(heroNum)][0] == teamName)
    if showAverage:
        teamAverage = np.zeros(shape=(dataframe.shape[0],))
    for heroNum in range(10):
        if dataframe["heroTeam" + str(heroNum)][0] == teamName:
            currentName = dataframe["heroName" + str(heroNum)][0].split("CDOTA_Unit_Hero_")
            currentName = currentName[1] if len(currentName) > 1 else currentName[0]
            currentHeroData = dataframe[statName + str(heroNum)]
            if showRaw:
                plt.plot(dataframe["tick"], currentHeroData, color=heroCols[currentHero], label=currentName + " " + statName)
            if showDerivs or showPolys:
                heroPoly = np.polynomial.Polynomial.fit(dataframe["tick"], currentHeroData, polyDegrees[currentHero])
                if showPolys:
                    heroPolyXs, heroPolyYs = heroPoly.linspace()
                    if rescalePolys:
                        plt.plot(heroPolyXs, (heroPolyYs / max(heroPolyYs)) * rescaleCoef, ":", color=heroCols[currentHero], label="Curve of best fit")
                    else:
                        plt.plot(heroPolyXs, heroPolyYs, ":", color=heroCols[currentHero], label="Curve of best fit")
                if showDerivs:
                    heroPolyDerivXs, heroPolyDerivYs = heroPoly.deriv().linspace()
                    if rescaleDerivs:
                        plt.plot(heroPolyDerivXs, (heroPolyDerivYs / max(heroPolyDerivYs)) * rescaleCoef, "--", color=heroCols[currentHero], label="Estimated rate of change")
                    else:
                        plt.plot(heroPolyDerivXs, heroPolyDerivYs, "--", color=heroCols[currentHero], label="Estimated rate of change")
            if showAverage:
                teamAverage += currentHeroData
            currentHero += 1
    
    if showAverage or showAveragePoly or showAverageDeriv:
        teamAverage /= 5
        if showAverage:
            if rescaleAverage:
                plt.plot(dataframe["tick"], (teamAverage / max(teamAverage)) * rescaleCoef, color=averageCol, label=teamName + " average " + statName)
            else:
                plt.plot(dataframe["tick"], teamAverage, color=averageCol, label=teamName + " average " + statName)
        if showAveragePoly:
            averagePoly = np.polynomial.Polynomial.fit(dataframe["tick"], teamAverage, averagePolyDegree)
            if showAveragePoly:
                averagePolyXs, averagePolyYs = averagePoly.linspace()
                if rescaleAveragePoly:
                    plt.plot(averagePolyXs, (averagePolyYs / max(averagePolyYs)) * rescaleCoef, ":", color=averageCol, label="Curve of best fit")
                else:
                    plt.plot(averagePolyXs, averagePolyYs, ":", color=averageCol, label="Curve of best fit")
            if showAverageDeriv:
                averageDerivXs, averageDerivYs = averagePoly.deriv().linspace()
                if rescaleAverageDeriv:
                    plt.plot(averageDerivXs, (averageDerivYs / max(averageDerivYs)) * rescaleCoef, "--", color=averageCol, label="Estimated rate of change")
                else:
                    plt.plot(averageDerivXs, averageDerivYs, "--", color=averageCol, label="Estimated rate of change")

    plt.xlabel("tick")
    plt.ylabel("total " + statName)
    plt.legend(loc='upper left')
    # print(plt.ylim())
    # print(plt.xlim())
    # plt.ylim(-21.8, 479.8)
    # plt.xlim(19781.55, 88211.45)
    plt.show()


# def cmpAllPlayersStat(statName, radPolyDegrees=[3, 3, 3, 3, 3], dirPolyDegrees=[3, 3, 3, 3, 3], radAveragePolyDegree=3, radHeroCols=['g', 'b', 'r', 'c', 'm'], averageCol='y', showPolys=False, rescalePolys=False, rescaleDerivs=True, showDerivs=True, showAverage=True, rescaleAverage=False, showAveragePoly=False, rescaleAveragePoly=False, showAverageDeriv=False, rescaleAverageDeriv=True):
#     global dataframe
#     global unitY
#     plt.clf()
#     dataframe = dataframe.sort_values("tick")
#     currentHero = 0
#     if (showPolys or showDerivs) and (rescalePolys or rescaleDerivs):
#         rescaleCoef = max(max(dataframe[statName + str(heroNum)]) for heroNum in range(10) if dataframe["heroTeam" + str(heroNum)][0] == teamName)
#     if showAverage:
#         teamAverage = np.zeros(shape=(dataframe.shape[0],))
#     for heroNum in range(10):
#         if dataframe["heroTeam" + str(heroNum)][0] == teamName:
#             currentName = dataframe["heroName" + str(heroNum)][0].split("CDOTA_Unit_Hero_")
#             currentName = currentName[1] if len(currentName) > 1 else currentName[0]
#             currentHeroData = dataframe[statName + str(heroNum)]
#             if showPolys:
#                 if rescalePolys:
#                     plt.plot(dataframe["tick"], (currentHeroData / max(currentHeroData)) * rescaleCoef, color=heroCols[currentHero], label=currentName + " " + statName)
#                 else:
#                     plt.plot(dataframe["tick"], currentHeroData, color=heroCols[currentHero], label=currentName + " " + statName)
#             if showDerivs or showPolys:
#                 heroPoly = np.polynomial.Polynomial.fit(dataframe["tick"], currentHeroData, polyDegrees[currentHero])
#                 if showPolys:
#                     heroPolyXs, heroPolyYs = heroPoly.linspace()
#                     if rescalePolys:
#                         plt.plot(heroPolyXs, (heroPolyYs / max(heroPolyYs)) * rescaleCoef, ":", color=heroCols[currentHero], label="Curve of best fit")
#                     else:
#                         plt.plot(heroPolyXs, heroPolyYs, ":", color=heroCols[currentHero], label="Curve of best fit")
#                 if showDerivs:
#                     heroPolyDerivXs, heroPolyDerivYs = heroPoly.deriv().linspace()
#                     if rescaleDerivs:
#                         plt.plot(heroPolyDerivXs, (heroPolyDerivYs / max(heroPolyDerivYs)) * rescaleCoef, "--", color=heroCols[currentHero], label="Estimated rate of change")
#                     else:
#                         plt.plot(heroPolyDerivXs, heroPolyDerivYs, "--", color=heroCols[currentHero], label="Estimated rate of change")
#             if showAverage:
#                 teamAverage += currentHeroData
#             currentHero += 1
    
#     if showAverage or showAveragePoly or showAverageDeriv:
#         teamAverage /= 5
#         if showAverage:
#             if rescaleAverage:
#                 plt.plot(dataframe["tick"], (teamAverage / max(teamAverage)) * rescaleCoef, color=averageCol, label=teamName + " average " + statName)
#             else:
#                 plt.plot(dataframe["tick"], teamAverage, color=averageCol, label=teamName + " average " + statName)
#         if showAveragePoly:
#             averagePoly = np.polynomial.Polynomial.fit(dataframe["tick"], teamAverage, averagePolyDegree)
#             if showAveragePoly:
#                 averagePolyXs, averagePolyYs = averagePoly.linspace()
#                 if rescaleAverage:
#                     plt.plot(averagePolyXs, (averagePolyYs / max(averagePolyYs)) * rescaleCoef, ":", color=averageCol, label="Curve of best fit")
#                 else:
#                     plt.plot(averagePolyXs, averagePolyYs, ":", color=averageCol, label="Curve of best fit")
#             if showAverageDeriv:
#                 averageDerivXs, averageDerivYs = averagePoly.deriv().linspace()
#                 if rescaleAverage:
#                     plt.plot(averageDerivXs, (averageDerivYs / max(averageDerivYs)) * rescaleCoef, "--", color=averageCol, label="Estimated rate of change")
#                 else:
#                     plt.plot(averageDerivXs, averageDerivYs, "--", color=averageCol, label="Estimated rate of change")

#     plt.xlabel("tick")
#     plt.ylabel("total " + statName)
#     plt.legend(loc='upper left')
#     plt.show()


def heatMapHeroStat(statName1, statName2, heroNum, bins=30, quantizeColour=False, colourBins=6, colours="viridis", showColourBar=True, showBarTicks=True):
    global dataframe
    plt.clf()
    dataframe = dataframe.sort_values("tick")
    # make a custom colormap with transparency
    # ncolors = 256
    # color_array = plt.get_cmap('YlOrRd')(range(ncolors))
    # color_array[:, -1] = np.linspace(0, 1, ncolors)
    # cmap = mcolors.LinearSegmentedColormap.from_list(name='YlOrRd_alpha', colors=color_array)
    heroName = dataframe["heroName" + str(heroNum)][0].split("CDOTA_Unit_Hero_")
    heroName = heroName[1] if len(heroName) > 1 else heroName[0]
    plt.xlabel(heroName + " " + statName1)
    plt.ylabel(heroName + " " + statName2)
    plt.hist2d(dataframe[statName1 + str(heroNum)], dataframe[statName2 + str(heroNum)], bins=[bins, bins], cmap=plt.cm.get_cmap(colours, colourBins) if quantizeColour else colours)
    if showColourBar:
        cbar = plt.colorbar(label="Time at position (ticks)")
        if not showBarTicks:
            cbar.set_ticks([])
    plt.show()




def radMapControlIsTurtle():
    global dataframe
    global unitY
    plt.clf()
    dataframe = dataframe.sort_values("mapControlRadiant")
    # plt.scatter(dataframe["mapControlRadiant"], unitY, s=1)
    for heroNum in range(10):
        plt.clf()
        line = plt.plot(dataframe["mapControlRadiant"], dataframe["isTurtling" + str(heroNum)], color=randomColour())
        plt.legend(line, "hero " + str(heroNum))
        plt.xlabel("mapControlRadiant")
        plt.ylabel("isTurtle")
        plt.show()


def radMapControlTurtlePred():
    global dataframe
    global unitY
    plt.clf()
    dataframe = dataframe.sort_values("mapControlRadiant")
    for heroNum in range(10):
        plt.clf()
        line = plt.plot(dataframe["mapControlRadiant"], dataframe["predictedIsTurtling" + str(heroNum)], color=randomColour())
        plt.legend(line, "hero " + str(heroNum))
        plt.xlabel("mapControlRadiant")
        plt.ylabel("isTurtle")
        plt.show()



# radMapControlIsTurtle()
# radMapControlTurtlePred()
# cmpMapControls()
# plotMapControls()

# Combine charts for both teams into a single plot, the same as what dotabuff does for team net worth advantage.
if action == "cmp":
    cmpTeamsStat(stat, 5, showRaw=True, showPoly=False, showDeriv=True, rescaleDerivs=True)
# Plot a stat for all members of one team on the same graph
elif action == "cmpteam":
    cmpWholeTeamStat(stat, sys.argv[4], polyDegrees=[3, 3, 3, 3, 3])
# Plot graphs for a stat for both teams, on the same chart
elif action == "plot":
    plotTeamsStat(stat, 5, dirDegree=4, showPolys=False, showDeriv=True, rescaleDerivs=True)
# Heatmap! cool with rates of position change
elif action == "heatmap":
    heatMapHeroStat(stat, sys.argv[4], sys.argv[5], showBarTicks=True)
# Compare a hero's stats to the average for the rest of their team
else:
    cmpHeroTeamStat(stat, sys.argv[4], False, 5, showPolys=False, teamDegree=3, average=True, rescaleDerivs=True, showDeriv=True)


# 3356083896-vitality_empire
# plotTeamsStat("netWorth", 7, showPolys=False)
# cmpTeamsStat("netWorth", 5, showRaw=True, showPoly=False, showDeriv=False)
# cmpHeroTeamStat("netWorth", 7, False, 5, showPolys=False, teamDegree=3)

# plotTeamsStat("XP", 2, showPolys=False)
# cmpTeamsStat("XP", 5, showRaw=True, showPoly=False, showDeriv=False)
# cmpHeroTeamStat("XP", 6, False, 3, showPolys=False, teamDegree=3, average=True)



# 3372676225-liquid_newbee
# plotTeamsStat("netWorth", 6, showPolys=False)
# cmpTeamsStat("netWorth", 5, showRaw=True, showPoly=False, showDeriv=False)
# cmpHeroTeamStat("netWorth", 9, False, 6, showPolys=False, teamDegree=2, average=True)

# plotTeamsStat("XP", 3, showPolys=False, dirDegree=2)
# cmpTeamsStat("XP", 5, showRaw=True, showPoly=False, showDeriv=False)
# cmpHeroTeamStat("XP", 9, False, 2, showPolys=False, teamDegree=2, average=True)
