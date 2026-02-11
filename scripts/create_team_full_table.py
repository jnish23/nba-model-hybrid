import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DB_PATH = r'c:\Users\jorda\nba-model-hybrid\data\nba_stats.db'

def create_full_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    logging.info("Dropping existing team_game_full table...")
    cur.execute("DROP TABLE IF EXISTS team_game_full")
    
    logging.info("Creating team_game_full table...")
    # Base: team_basic_stats
    # Joins: All others
    
    query = """
    CREATE TABLE team_game_full AS
    SELECT 
        -- Basic
        tb.seasonYear,
        tb.teamId,
        tb.teamTricode,
        tb.gameId,
        tb.gameDate,
        tb.matchup,
        tb.wl,
        tb.minutes,
        tb.points,
        tb.plusMinus,
        tb.fieldGoalsMade,
        tb.fieldGoalsAttempted,
        tb.fieldGoalPercentage,
        tb.threePointersMade,
        tb.threePointersAttempted,
        tb.threePointerPercentage,
        tb.freeThrowsMade,
        tb.freeThrowsAttempted,
        tb.freeThrowPercentage,
        tb.offensiveRebounds,
        tb.defensiveRebounds,
        tb.rebounds,
        tb.assists,
        tb.steals,
        tb.blocks,
        tb.turnovers,
        tb.personalFouls,
        
        -- Advanced
        ta.offensiveRating,
        ta.defensiveRating,
        ta.netRating,
        ta.assistPercentage,
        ta.assistToTurnover,
        ta.assistRatio,
        ta.offensiveReboundPercentage,
        ta.defensiveReboundPercentage,
        ta.reboundPercentage,
        ta.effectiveFieldGoalPercentage,
        ta.trueShootingPercentage,
        ta.usagePercentage,
        ta.pace,
        ta.pie,
        
        -- Scoring
        ts.percentageFieldGoalsAttempted2pt, 
        ts.percentageFieldGoalsAttempted3pt, 
        ts.percentagePoints2pt, 
        ts.percentagePointsMidrange2pt, 
        ts.percentagePoints3pt, 
        ts.percentagePointsFastBreak, 
        ts.percentagePointsFreeThrow, 
        ts.percentagePointsOffTurnovers, 
        ts.percentagePointsPaint, 
        ts.percentageAssisted2pt, 
        ts.percentageUnassisted2pt, 
        ts.percentageAssisted3pt, 
        ts.percentageUnassisted3pt, 
        ts.percentageAssistedFGM, 
        ts.percentageUnassistedFGM,
        
        -- Track
        tt.distance,
        tt.speed,
        tt.touches,
        tt.secondaryAssists,
        tt.freeThrowAssists,
        tt.passes,
        tt.contestedFieldGoalsMade,
        tt.contestedFieldGoalsAttempted,
        tt.contestedFieldGoalPercentage,
        tt.uncontestedFieldGoalsMade,
        tt.uncontestedFieldGoalsAttempted,
        tt.uncontestedFieldGoalsPercentage,
        tt.defendedAtRimFieldGoalsMade,
        tt.defendedAtRimFieldGoalsAttempted,
        tt.defendedAtRimFieldGoalPercentage,
        tt.reboundChancesOffensive,
        tt.reboundChancesDefensive,
        tt.reboundChancesTotal,
        
        -- Hustle
        th.contestedShots,
        th.contestedShots2pt,
        th.contestedShots3pt,
        th.deflections,
        th.chargesDrawn,
        th.screenAssists,
        th.screenAssistPoints,
        th.looseBallsRecoveredOffensive,
        th.looseBallsRecoveredDefensive,
        th.looseBallsRecoveredTotal,
        th.offensiveBoxOuts,
        th.defensiveBoxOuts,
        th.boxOutPlayerTeamRebounds,
        th.boxOutPlayerRebounds,
        th.boxOuts,

        -- Shot Location (Join on Date/Team)
        tsl."Restricted Area FGM", tsl."Restricted Area FGA", tsl."Restricted Area FG_PCT",
        tsl."In The Paint (Non-RA) FGM", tsl."In The Paint (Non-RA) FGA", tsl."In The Paint (Non-RA) FG_PCT",
        tsl."Mid-Range FGM", tsl."Mid-Range FGA", tsl."Mid-Range FG_PCT",
        tsl."Left Corner 3 FGM", tsl."Left Corner 3 FGA", tsl."Left Corner 3 FG_PCT",
        tsl."Right Corner 3 FGM", tsl."Right Corner 3 FGA", tsl."Right Corner 3 FG_PCT",
        tsl."Above the Break 3 FGM", tsl."Above the Break 3 FGA", tsl."Above the Break 3 FG_PCT",
        tsl."Backcourt FGM", tsl."Backcourt FGA", tsl."Backcourt FG_PCT",
        tsl."Corner 3 FGM", tsl."Corner 3 FGA", tsl."Corner 3 FG_PCT",

        -- Defense (Join on Date/Team)
        dp.* -- Select all for now, relying on unique names or handle specific cols if collision
             -- Logic: dp columns are distinct enough (e.g. OPP_FGM_C)
        
    FROM team_basic_stats tb
    
    LEFT JOIN team_advanced_stats_v3 ta ON tb.teamId = ta.teamId AND tb.gameId = ta.gameId
    LEFT JOIN team_scoring_stats_v3 ts ON tb.teamId = ts.teamId AND tb.gameId = ts.gameId
    LEFT JOIN team_track_stats_v3 tt ON tb.teamId = tt.teamId AND tb.gameId = tt.gameId
    LEFT JOIN team_hustle_stats th ON tb.teamId = th.teamId AND tb.gameId = th.gameId
    
    -- Date-based joins
    LEFT JOIN team_shot_location_boxscores tsl ON tb.teamId = tsl.teamId AND tb.gameDate = tsl.gameDate
    LEFT JOIN defensive_stats_by_position dp ON tb.teamId = dp.teamId AND tb.gameDate = dp.gameDate
    
    """
    
    # Prune duplicate columns from 'dp' selection if necessary? 
    # dp has teamId, teamName, gameDate, season. We don't want duplicates of those.
    # So instead of dp.*, let's list them or just ignore collisions (SQLite handles them by suffixing usually: season:1)
    
    # Refined Query without dp.* for safety
    
    safe_query = """
    CREATE TABLE team_game_full AS
    SELECT 
        tb.seasonYear, tb.teamId, tb.teamTricode, tb.gameId, tb.gameDate, tb.matchup, tb.wl,
        
        -- Basic Stats
        tb.minutes, tb.points, tb.plusMinus, tb.fieldGoalsMade, tb.fieldGoalsAttempted, tb.fieldGoalPercentage,
        tb.threePointersMade, tb.threePointersAttempted, tb.threePointerPercentage,
        tb.freeThrowsMade, tb.freeThrowsAttempted, tb.freeThrowPercentage,
        tb.offensiveRebounds, tb.defensiveRebounds, tb.rebounds, tb.assists, tb.steals, tb.blocks, tb.turnovers, tb.personalFouls,

        -- Advanced
        ta.offensiveRating, ta.defensiveRating, ta.netRating, ta.assistPercentage, ta.assistToTurnover, ta.assistRatio,
        ta.offensiveReboundPercentage, ta.defensiveReboundPercentage, ta.reboundPercentage, ta.effectiveFieldGoalPercentage,
        ta.trueShootingPercentage, ta.usagePercentage, ta.pace, ta.pie,

        -- Scoring
        ts.percentageFieldGoalsAttempted2pt, ts.percentageFieldGoalsAttempted3pt, ts.percentagePoints2pt, ts.percentagePointsMidrange2pt,
        ts.percentagePoints3pt, ts.percentagePointsFastBreak, ts.percentagePointsFreeThrow, ts.percentagePointsOffTurnovers,
        ts.percentagePointsPaint, ts.percentageAssisted2pt, ts.percentageUnassisted2pt, ts.percentageAssisted3pt,
        ts.percentageUnassisted3pt, ts.percentageAssistedFGM, ts.percentageUnassistedFGM,

        -- Track
        tt.distance, tt.speed, tt.touches, tt.secondaryAssists, tt.freeThrowAssists, tt.passes,
        tt.contestedFieldGoalsMade, tt.contestedFieldGoalsAttempted, tt.contestedFieldGoalPercentage,
        tt.uncontestedFieldGoalsMade, tt.uncontestedFieldGoalsAttempted, tt.uncontestedFieldGoalsPercentage,
        tt.defendedAtRimFieldGoalsMade, tt.defendedAtRimFieldGoalsAttempted, tt.defendedAtRimFieldGoalPercentage,
        tt.reboundChancesOffensive, tt.reboundChancesDefensive, tt.reboundChancesTotal,

        -- Hustle
        th.contestedShots, th.contestedShots2pt, th.contestedShots3pt, th.deflections, th.chargesDrawn,
        th.screenAssists, th.screenAssistPoints, th.looseBallsRecoveredOffensive, th.looseBallsRecoveredDefensive,
        th.looseBallsRecoveredTotal, th.offensiveBoxOuts, th.defensiveBoxOuts, th.boxOutPlayerTeamRebounds,
        th.boxOutPlayerRebounds, th.boxOuts,

        -- Shot Location
        tsl."Restricted Area FGM", tsl."Restricted Area FGA", tsl."Restricted Area FG_PCT",
        tsl."In The Paint (Non-RA) FGM", tsl."In The Paint (Non-RA) FGA", tsl."In The Paint (Non-RA) FG_PCT",
        tsl."Mid-Range FGM", tsl."Mid-Range FGA", tsl."Mid-Range FG_PCT",
        tsl."Above the Break 3 FGM", tsl."Above the Break 3 FGA", tsl."Above the Break 3 FG_PCT",
        tsl."Corner 3 FGM", tsl."Corner 3 FGA", tsl."Corner 3 FG_PCT",

        -- Defense (explicitly listing to avoid duplicate join keys being selected)
        dp.OPP_FGM_C, dp.OPP_FGM_F, dp.OPP_FGM_G, dp.OPP_FGA_C, dp.OPP_FGA_F, dp.OPP_FGA_G,
        dp.OPP_FG_PCT_C, dp.OPP_FG_PCT_F, dp.OPP_FG_PCT_G, dp.OPP_FG3M_C, dp.OPP_FG3M_F, dp.OPP_FG3M_G,
        dp.OPP_FG3A_C, dp.OPP_FG3A_F, dp.OPP_FG3A_G, dp.OPP_FG3_PCT_C, dp.OPP_FG3_PCT_F, dp.OPP_FG3_PCT_G,
        dp.OPP_FTM_C, dp.OPP_FTM_F, dp.OPP_FTM_G, dp.OPP_FTA_C, dp.OPP_FTA_F, dp.OPP_FTA_G,
        dp.OPP_FT_PCT_C, dp.OPP_FT_PCT_F, dp.OPP_FT_PCT_G, dp.OPP_OREB_C, dp.OPP_OREB_F, dp.OPP_OREB_G,
        dp.OPP_DREB_C, dp.OPP_DREB_F, dp.OPP_DREB_G, dp.OPP_REB_C, dp.OPP_REB_F, dp.OPP_REB_G,
        dp.OPP_AST_C, dp.OPP_AST_F, dp.OPP_AST_G, dp.OPP_TOV_C, dp.OPP_TOV_F, dp.OPP_TOV_G,
        dp.OPP_STL_C, dp.OPP_STL_F, dp.OPP_STL_G, dp.OPP_BLK_C, dp.OPP_BLK_F, dp.OPP_BLK_G,
        dp.OPP_BLKA_C, dp.OPP_BLKA_F, dp.OPP_BLKA_G, dp.OPP_PF_C, dp.OPP_PF_F, dp.OPP_PF_G,
        dp.OPP_PFD_C, dp.OPP_PFD_F, dp.OPP_PFD_G, dp.OPP_PTS_C, dp.OPP_PTS_F, dp.OPP_PTS_G,
        dp.PLUS_MINUS_C AS PLUS_MINUS_C_DEF, dp.PLUS_MINUS_F AS PLUS_MINUS_F_DEF, dp.PLUS_MINUS_G AS PLUS_MINUS_G_DEF
        
    FROM team_basic_stats tb
    
    LEFT JOIN team_advanced_stats_v3 ta ON tb.teamId = ta.teamId AND tb.gameId = ta.gameId
    LEFT JOIN team_scoring_stats_v3 ts ON tb.teamId = ts.teamId AND tb.gameId = ts.gameId
    LEFT JOIN team_track_stats_v3 tt ON tb.teamId = tt.teamId AND tb.gameId = tt.gameId
    LEFT JOIN team_hustle_stats th ON tb.teamId = th.teamId AND tb.gameId = th.gameId
    
    LEFT JOIN team_shot_location_boxscores tsl ON tb.teamId = tsl.teamId AND tb.gameDate = tsl.gameDate
    LEFT JOIN defensive_stats_by_position dp ON tb.teamId = dp.teamId AND tb.gameDate = dp.gameDate
    """
    
    try:
        cur.execute(safe_query)
        rows = cur.execute("SELECT COUNT(*) FROM team_game_full").fetchone()[0]
        logging.info(f"Table created successfully with {rows} rows.")
        
        logging.info("Indexing...")
        cur.execute("CREATE INDEX idx_full_team_date ON team_game_full(teamId, gameDate)")
        cur.execute("CREATE INDEX idx_full_game_id ON team_game_full(gameId)")
        
        conn.commit()
    except Exception as e:
        logging.error(f"Error creating table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_full_table()
