An example script for downloading DOTA 2 demo files using the OpenDota API.

For data comparability, my project only used data from a single version of DOTA 2; Version 7.06e.

To obtain match IDs from a single version of DOTA 2, I made a query to the [OpenDota data explorer](https://www.opendota.com/explorer).

The SQL query is provided below. To be taken to the results of the query, click [here](https://api.opendota.com/api/explorer?sql=SELECT%20matches.match_id%0D%0AFROM%20matches%0D%0AWHERE%20matches.start_time%20%3E%3D%20extract(epoch%20from%20timestamp%20%272017-07-02%27)%0D%0AAND%20matches.start_time%20%3C%3D%20extract(epoch%20from%20timestamp%20%272017-08-20%27)).

```
SELECT matches.match_id
FROM matches
WHERE matches.start_time >= extract(epoch from timestamp '2017-07-02')
AND matches.start_time <= extract(epoch from timestamp '2017-08-20')
```