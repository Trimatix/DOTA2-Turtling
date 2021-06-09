import requests, json, time, os
from typing import List, Dict
import bz2

BS2DIR = "data\download"
DEMDIR = "data\extract"
CSVDIR = "data\parse"
DELETE_BS2 = True
DELETE_DEM = False


def getURLParams(matchID: int) -> Dict[str, int]:
    request = requests.get(f"https://api.opendota.com/api/replays?match_id={matchID!s}")
    if request.ok:
        # print("GET replay URL params:", matchID)
        return request.json()[0]
    raise RuntimeError(f"request not ok. Error {request.status_code}")

def downloadReplay(urlParams: Dict[str, int], folder: str) -> str:
    request = requests.get(f"http://replay{urlParams['cluster']}.valve.net/570/{urlParams['match_id']}_{urlParams['replay_salt']}.dem.bz2")
    if request.ok:
        # print("GET replay file:", urlParams['match_id'])
        with open(folder + os.sep + f"{urlParams['match_id']}.dem.bz2", 'wb') as f:
            f.write(request.content)
        return folder + os.sep + f"{urlParams['match_id']}.dem.bz2"
    raise RuntimeError(f"request not ok. Error {request.status_code}")


def decompressBZ2(filepath: str, outfolder: str) -> str:
    fname = os.path.basename(filepath)
    zipfile = bz2.BZ2File(filepath) # open the file
    data = zipfile.read() # get the decompressed data
    newfilepath = os.path.join(outfolder, fname[:-4]) # assuming the filepath ends with .bz2
    with open(newfilepath, 'wb') as f: # write a uncompressed file
        f.write(data)
    return newfilepath
        

for path in (BS2DIR, DEMDIR, CSVDIR):
    if not os.path.isdir(path):
        os.makedirs(path)


with open("match IDs.json", "r") as f:
    matchesData = json.load(f)
currentMatchID = -1
numMatches = matchesData["rowCount"]//10 # assumes 10 players per match
downloadedMatches = 0
for matchData in matchesData["rows"]:
    if matchData["match_id"] != currentMatchID:
        currentMatchID = matchData["match_id"]
        bs2Fname = f"{currentMatchID}.dem.bz2"
        bs2Path = os.path.join(BS2DIR, bs2Fname)
        demFname = f"{currentMatchID}.dem"
        demPath = os.path.join(DEMDIR, demFname)

        if os.path.isfile(bs2Path):
            # if not os.path.isfile(demPath):
            print("match",currentMatchID,"already downloaded, skipping")
            downloadedMatches += 1
            if not os.path.isfile(demPath):
                try:
                    decompressBZ2(bs2Path, DEMDIR)
                except OSError as e:
                    print(f"Exception when unzipping replay file '{replayFile}': {str(e)}")
            if DELETE_BS2:
                os.remove(bs2Path)
        if not os.path.isfile(demPath):
            progress = "%.1f" % (((downloadedMatches+1)/numMatches)*100)
            print(f"match {downloadedMatches+1}/{numMatches} ({progress}%): {currentMatchID}")
            try:
                urlParams = getURLParams(currentMatchID)
                time.sleep(2)
            except RuntimeError as e:
                print(f"Exception when getting URL params for match {currentMatchID}: {e.args}")
            else:
                try:
                    replayFile = downloadReplay(urlParams, BS2DIR)
                    time.sleep(2)
                except RuntimeError as e:
                    print(f"Exception when downloading replay file for params '{urlParams}': {e.args}")
                else:
                    downloadedMatches += 1
                    try:
                        decompressBZ2(replayFile, DEMDIR)
                    except OSError as e:
                        print(f"Exception when unzipping replay file '{replayFile}': {str(e)}")
                    else:
                        if DELETE_BS2:
                            os.remove(bs2Path)

print(f"\n\nDownloading complete, fetched {downloadedMatches} matches.")

# startID = 3372676377
# # startID = 3372676225
# numMatches = 1848
# for i in range(numMatches):
#     getMatchByID(startID + i)
#     time.sleep(2)
    