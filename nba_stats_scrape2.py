import pandas as pd
import numpy as np

from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.endpoints import boxscoreadvancedv2

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import sys

start_time = time.time()

year = sys.argv[1]

fails_name = 'fails' + year + '.csv'
result_name = 'result' + year + '.csv'

start_year = 2000 + int(year)
end_year = 2000 + int(year)

print('Processing year ', start_year)

games = []

for season_id in range(start_year,end_year+1):
    seasongames = leaguegamelog.LeagueGameLog(season=str(season_id), timeout=15).get_data_frames()[0]
    games += list(seasongames['GAME_ID'].unique())

print('Successfully got game ids!')

game_count = len(games)
print('Game count ', game_count)

df_games = pd.DataFrame(games, columns=['Games'])
df_games.to_csv('games.csv', index=False)

# games = games[0:101]

max_attempts = 3
failed_games = []
count = 0

def addGame(game_id, attempts=0):
    try:
        data_frame = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id,timeout=10).get_data_frames()[0]

        global count
        count += 1
        if count % 100 == 0:
            print('Successes ', count, ' out of ', game_count)

        return data_frame
    except Exception as e:
        if attempts >= max_attempts - 1:
            print("Couldn't add game ", game_id)
            failed_games.append(game_id)
            return None

        time.sleep(2 ** attempts)
        return addGame(game_id, attempts+1)

df = pd.DataFrame()

with ThreadPoolExecutor(max_workers=16) as executor:
    print('Max workers = ', executor._max_workers)

    these_futures = [executor.submit(addGame, game) for game in games]

    for future in as_completed(these_futures):
        result = future.result()
        if result is not None:
            df = pd.concat([df, result], ignore_index=True)


print('Finished processing')
df.to_csv(result_name, index=False)

df_fails = pd.DataFrame(failed_games, columns=['Failed Games'])
df_fails.to_csv(fails_name, index=False)

print("--- %s seconds ---" % (time.time() - start_time))