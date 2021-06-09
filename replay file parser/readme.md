Dota 2 replay file parser, written with [Clarity](https://github.com/skadistats/clarity).

Data labelling columns must be added to the ends of your CSVs (`isTurtling0`, `isTurtling`, ... `isTurtling9`) before they will be useable with the model.

A `heroes.json` file must be present alongside your compiled binary, like what can be obtained [from opendota](https://api.opendota.com/api/constants/heroes). An example `heroes.json` (provided by the aforementioned) can be found in the target folder.